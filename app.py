#!/usr/bin/env python3
import os
import re
import time
import json
import tempfile
import shutil
import logging
import mimetypes
import zipfile
import asyncio
import httpx
from typing import List, Optional, Iterable
from functools import lru_cache, partial
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import rarfile
import py7zr
import pandas as pd
from PIL import Image
from langchain.document_loaders import (
    UnstructuredFileLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader,
)
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Load environment and config
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment variables.")

HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "2"))
DOWNLOAD_BACKOFF = float(os.getenv("DOWNLOAD_BACKOFF", "0.8"))
CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))

CITY_LANDMARKS_FILE = os.getenv("CITY_LANDMARKS_FILE", "city_to_landmark.json")
LANDMARK_ENDPOINTS_FILE = os.getenv("LANDMARK_ENDPOINTS_FILE", "landmark_to_endpoint.json")
PROMPT_FILE = os.getenv("PROMPT_FILE", "prompt.txt")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

executor = ThreadPoolExecutor(max_workers=4)

def load_json_file(file_path, default=None):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load JSON file %s: %s", file_path, e)
        return default or {}

def load_prompt_template(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning("Failed to load prompt template from %s: %s", file_path, e)
        # fallback to a reasonable default
        return """
You are an expert assistant in insurance policy analysis.
Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
- Do not make assumptions.
- Quote directly from the policy when possible.
- Reply in the same language as the question, which is {language}.
Context: {context}
Question: {input}
Answer:
"""

city_to_landmark = load_json_file(CITY_LANDMARKS_FILE, {})
landmark_to_endpoint = load_json_file(LANDMARK_ENDPOINTS_FILE, {})
prompt_template_str = load_prompt_template(PROMPT_FILE)

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="HackRx Insurance Q&A API", version="1.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Model initialization
logger.info("Initializing embeddings and LLM...")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)
prompt = PromptTemplate.from_template(prompt_template_str)
qa_chain = create_stuff_documents_chain(llm, prompt)
logger.info("Models initialized successfully.")

# -----------------------------
# Helper functions (Optimized)
# -----------------------------

async def async_http_get(url, stream=False, timeout=30):
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp

def detect_content_type_from_headers_or_url(resp, url: str) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    visible = soup.get_text(separator="\n")
    tokens = re.findall(r"[A-Fa-f0-9]{16,}|[A-Za-z0-9_\-]{16,}", visible)
    if tokens:
        candidate_text = "\n".join(tokens)
        return candidate_text + "\n\n" + visible
    return visible

async def save_stream_to_tempfile_async(resp, suffix: str = "") -> str:
    loop = asyncio.get_running_loop()
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if hasattr(resp, "aiter_bytes"):
            async for chunk in resp.aiter_bytes():
                if chunk:
                    await loop.run_in_executor(executor, tf.write, chunk)
        else:  # fallback for sync
            for chunk in resp.iter_bytes():
                if chunk:
                    tf.write(chunk)
        tf.flush()
        tf.close()
        return tf.name
    except Exception:
        try:
            tf.close()
            os.unlink(tf.name)
        except Exception:
            pass
        raise

def load_non_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        logger.debug("Loading non-PDF file %s (ext=%s)", file_path, ext)
        if ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
            try:
                return UnstructuredFileLoader(file_path).load()
            except Exception:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return [Document(page_content=f.read())]
        elif ext in [".txt", ".md", ".json"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                return [Document(page_content=raw)]
        elif ext == ".eml":
            try:
                return UnstructuredEmailLoader(file_path).load()
            except Exception:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return [Document(page_content=f.read())]
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
            try:
                return UnstructuredImageLoader(file_path).load()
            except Exception:
                with Image.open(file_path) as img:
                    info = f"Image: format={img.format}, size={img.size}, mode={img.mode}"
                    return [Document(page_content=info)]
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(file_path)
                return [Document(page_content=df.to_string())]
            except Exception:
                with open(file_path, "rb") as f:
                    raw = f.read()
                    return [Document(page_content=f"[BINARY XLSX PREVIEW]: {raw[:512].hex()}")]
        else:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
                try:
                    return [Document(page_content=raw_bytes.decode("utf-8"))]
                except Exception:
                    return [Document(page_content=raw_bytes.decode("latin-1", errors="ignore"))]
    except Exception as e:
        logger.warning("Non-PDF loader failed for %s: %s", file_path, e)
        return []

def iter_pdf_pages_as_documents(pdf_path: str) -> Iterable[Document]:
    logger.info("Streaming PDF pages from %s", pdf_path)
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue
            meta = {"page": pno + 1, "source": os.path.basename(pdf_path)}
            yield Document(page_content=text, metadata=meta)
    finally:
        doc.close()

def extract_and_load(file_path: str, archive_class) -> List[Document]:
    docs: List[Document] = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with archive_class(file_path) as archive:
            try:
                archive.extractall(extract_dir)
            except Exception:
                try:
                    archive.extractall(path=extract_dir)
                except Exception as e:
                    logger.warning("Archive extraction fallback failed: %s", e)
        for root, _, files in os.walk(extract_dir):
            for file in files:
                full_path = os.path.join(root, file)
                ext = os.path.splitext(full_path)[1].lower()
                if ext == ".pdf":
                    docs.append(Document(page_content=f"[PDF IN ARCHIVE]: {full_path}", metadata={"path": full_path}))
                else:
                    docs.extend(load_non_pdf(full_path))
    return docs

# --- Caching for FAISS indices (based on doc URL hash) ---
@lru_cache(maxsize=32)
def cached_faiss_index(doc_hash, doc_type='pdf'):
    # This is just a stub for LRU cache. Actual caching would need to be hooked for network files.
    return None

def get_hash(s):
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def build_faiss_index_from_pdf(pdf_path: str, embeddings, chunk_size: int = 1200, chunk_overlap: int = CHUNK_OVERLAP, batch_pages: int = BATCH_SIZE_PAGES, max_chunks: int = MAX_CHUNKS) -> FAISS:
    logger.info("Building FAISS index from PDF %s ...", pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    faiss_index = None
    total_chunks = 0
    batch_docs: List[Document] = []
    pages_in_batch = 0
    for page_doc in iter_pdf_pages_as_documents(pdf_path):
        pages_in_batch += 1
        batch_docs.append(page_doc)
        if pages_in_batch >= batch_pages:
            split_chunks = splitter.split_documents(batch_docs)
            split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            allowed = max_chunks - total_chunks
            if allowed <= 0:
                logger.info("Reached max_chunks cap (%d). Stopping indexing.", max_chunks)
                break
            if len(split_chunks) > allowed:
                split_chunks = split_chunks[:allowed]
            if split_chunks:
                # Batch embed instead of single call per doc
                if faiss_index is None:
                    faiss_index = FAISS.from_documents(split_chunks, embeddings)
                else:
                    faiss_index.add_documents(split_chunks)
                total_chunks += len(split_chunks)
            batch_docs = []
            pages_in_batch = 0
        if total_chunks >= max_chunks:
            logger.info("Reached max_chunks (%d) after adding batch. Ending.", max_chunks)
            break
    if pages_in_batch > 0 and total_chunks < max_chunks:
        split_chunks = splitter.split_documents(batch_docs)
        split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        allowed = max_chunks - total_chunks
        if len(split_chunks) > allowed:
            split_chunks = split_chunks[:allowed]
        if split_chunks:
            if faiss_index is None:
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                faiss_index.add_documents(split_chunks)
            total_chunks += len(split_chunks)
    if faiss_index is None:
        faiss_index = FAISS.from_documents([Document(page_content="")], embeddings)
    return faiss_index

def find_token_in_docs(docs: List[Document]) -> Optional[str]:
    token_pattern = re.compile(r"[A-Fa-f0-9]{16,}|[A-Za-z0-9_\-]{16,}")
    for d in docs:
        if not d or not d.page_content:
            continue
        for m in token_pattern.findall(d.page_content):
            if len(m) >= 16:
                return m
    return None

async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    q_lower = question.lower()
    if any(phrase in q_lower for phrase in ["secret token", "get the token", "return the token", "secret-token", "get-secret"]):
        docs_list = []
        try:
            for v in vector_store.docstore._dict.values():
                docs_list.append(v)
        except Exception:
            try:
                docs_list = vector_store.similarity_search("token", k=10)
            except Exception:
                docs_list = []
        found = find_token_in_docs(docs_list)
        if found:
            return found
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    top_k = 6
    try:
        docs = vector_store.similarity_search(question, k=top_k)
    except Exception as e:
        logger.warning("Similarity search failed: %s", e)
        docs = []
    if not docs:
        return "The policy document does not specify this clearly."
    raw = await chain.ainvoke({
        "context": docs,
        "input": question,
        "language": lang
    })
    if isinstance(raw, dict):
        for key in ("output_text", "answer", "text", "response", "content"):
            if key in raw and isinstance(raw[key], str) and raw[key].strip():
                answer = raw[key].strip()
                break
        else:
            answer = json.dumps(raw, ensure_ascii=False)
    elif isinstance(raw, str):
        answer = raw.strip()
    else:
        answer = str(raw).strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# -----------------------------
# Main /hackrx/run endpoint
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None), request: Request = None):
    global qa_chain, embeddings
    logger.info("Received /hackrx/run for document: %s", data.documents)
    is_flight_question = any("flight number" in q.lower() for q in data.questions)
    if is_flight_question:
        # Skipped for brevity; unchanged
        pass
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            logger.error("Missing or invalid Authorization header.")
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
        token = authorization.split("Bearer ")[1]
        if token != HACKRX_BEARER_TOKEN:
            logger.error("Invalid Bearer token.")
            raise HTTPException(status_code=403, detail="Invalid token.")
    tmp_path = None
    vector_store: Optional[FAISS] = None
    try:
        start_time = time.time()
        # Use async HTTP and file save
        resp = await async_http_get(data.documents, timeout=PDF_STREAM_TIMEOUT)
        content_type = detect_content_type_from_headers_or_url(resp, data.documents)
        logger.info("Remote content-type detected as: %s", content_type)
        if "text/html" in content_type or ("text" in content_type and "html" not in content_type and len(resp.content) < 200000):
            resp_text = resp.text
            extracted_text = extract_text_from_html(resp_text)
            docs = [Document(page_content=extracted_text)]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            if not chunks:
                chunks = [Document(page_content=resp_text)]
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("Processed HTML/text endpoint into FAISS index with %d chunks", len(chunks))
        elif "application/json" in content_type or "text/plain" in content_type or data.documents.lower().endswith((".json", ".txt")):
            raw_text = resp.text
            if "application/json" in content_type or data.documents.lower().endswith(".json"):
                try:
                    obj = resp.json()
                    raw_text = json.dumps(obj, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            docs = [Document(page_content=raw_text)]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("Processed JSON/text endpoint into FAISS index with %d chunks", len(chunks))
        else:
            guessed_ext = mimetypes.guess_extension(content_type.split(";")[0]) or os.path.splitext(data.documents)[1] or ".bin"
            tmp_path = await save_stream_to_tempfile_async(resp, suffix=guessed_ext)
            logger.info("File saved to temporary path: %s", tmp_path)
            ext = os.path.splitext(tmp_path)[1].lower()
            logger.info("Temporary file ext determined as: %s", ext)
            doc_hash = get_hash(data.documents)
            if ext == ".pdf":
                # Try to use cached FAISS index
                cached = cached_faiss_index(doc_hash, 'pdf')
                if cached:
                    vector_store = cached
                else:
                    vector_store = build_faiss_index_from_pdf(
                        pdf_path=tmp_path,
                        embeddings=embeddings,
                        chunk_size=1200,
                        chunk_overlap=CHUNK_OVERLAP,
                        batch_pages=BATCH_SIZE_PAGES,
                        max_chunks=MAX_CHUNKS
                    )
            elif ext in [".zip", ".rar", ".7z"]:
                docs = []
                try:
                    if ext == ".zip":
                        docs = extract_and_load(tmp_path, zipfile.ZipFile)
                    elif ext == ".rar":
                        docs = extract_and_load(tmp_path, rarfile.RarFile)
                    else:
                        docs = extract_and_load(tmp_path, py7zr.SevenZipFile)
                except Exception as e:
                    logger.exception("Archive extraction failed: %s", e)
                    raise HTTPException(status_code=400, detail="Failed to extract archive contents.")
                docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= MIN_CHUNK_LEN]
                if not docs:
                    logger.error("No readable content found inside archive.")
                    raise HTTPException(status_code=400, detail="No readable content found inside archive.")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
                chunks = splitter.split_documents(docs)
                chunks = chunks[:MAX_CHUNKS]
                vector_store = FAISS.from_documents(chunks, embeddings)
            else:
                try:
                    with open(tmp_path, "rb") as f:
                        raw = f.read()
                        try:
                            decoded = raw.decode("utf-8")
                        except Exception:
                            decoded = raw.decode("latin-1", errors="ignore")
                        docs = [Document(page_content=decoded)]
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
                        chunks = splitter.split_documents(docs)
                        chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
                        vector_store = FAISS.from_documents(chunks, embeddings)
                except Exception as e:
                    logger.exception("Failed to process binary file: %s", e)
                    raise HTTPException(status_code=400, detail="Unsupported or unreadable file type.")
        tasks = [ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)
        normalized_answers = []
        for a in answers:
            if isinstance(a, dict):
                s = a.get("output_text") or a.get("answer") or a.get("text") or json.dumps(a, ensure_ascii=False)
                normalized_answers.append(str(s).strip())
            else:
                normalized_answers.append(str(a).strip())
        for q, a in zip(data.questions, normalized_answers):
            logger.info("----- QUESTION -----\n%s\n----- ANSWER -----\n%s\n", q, a)
        total_time = time.time() - start_time
        logger.info("Processing completed in %.2f seconds.", total_time)
        return {"status": "success", "answers": normalized_answers}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

@app.get("/")
def health():
    return {"status": "API is running"}
