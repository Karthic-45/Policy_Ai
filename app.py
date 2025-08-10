#!/usr/bin/env python3
"""
app.py - HackRx Insurance Q&A API (full-featured)

Features:
- FastAPI endpoint POST /hackrx/run
- Accepts {"documents": "<url>", "questions": ["q1", "q2", ...]}
- Supports:
    - PDF URLs (streamed + page-by-page extraction via PyMuPDF)
    - Archive files (.zip, .rar, .7z) containing documents
    - Plain text and JSON endpoints (e.g. get-secret-token page)
    - HTML pages (extracts visible text and long token strings)
- Builds FAISS index incrementally for large PDFs (page-batched)
- Uses OpenAI embeddings (text-embedding-3-small) and ChatOpenAI LLM
- Uses a "stuff documents" qa_chain with a policy-specific prompt
- Bearer token authentication using env var HACKRX_BEARER_TOKEN (optional)
- Detailed logging and cleanup
- Concurrency: answers multiple questions concurrently
- Many configuration knobs via env vars
"""

import os
import re
import time
import json
import tempfile
import shutil
import logging
import mimetypes
import zipfile
from typing import List, Optional, Iterable, Tuple
from functools import partial

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import asyncio

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect

# LangChain / OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Archive libs
import rarfile
import py7zr

# -----------------------------
# Load environment and config
# -----------------------------
load_dotenv()

# Required env vars:
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment variables.")

# Optional config via env
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")  # If set, endpoint requires it.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "2"))
DOWNLOAD_BACKOFF = float(os.getenv("DOWNLOAD_BACKOFF", "0.8"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hackrx_app")

# Ensure OpenAI key available to underlying libs (some expect env var)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG + GPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request/Response models
# -----------------------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# -----------------------------
# Model initialization
# -----------------------------
try:
    logger.info("Initializing embeddings and LLM...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # uses env OPENAI_API_KEY
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)  # uses env OPENAI_API_KEY

    # Prompt template for policy QA (keeps language variable)
    prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
- Do not make assumptions.
- Quote directly from the policy when possible.
- Reply in the same language as the question, which is {language}.

Context:
{context}

Question: {input}
Answer:
""")

    qa_chain = create_stuff_documents_chain(llm, prompt)
    logger.info("Models initialized successfully.")
except Exception as e:
    logger.exception("Failed to initialize models: %s", e)
    raise

# -----------------------------
# Utility functions
# -----------------------------
def _retry_request(url: str, stream: bool = False, timeout: int = 30) -> requests.Response:
    """Retry wrapper for requests.get with simple backoff."""
    last_exc = None
    for attempt in range(1 + DOWNLOAD_RETRIES):
        try:
            logger.debug("Downloading (%d/%d): %s", attempt + 1, DOWNLOAD_RETRIES + 1, url)
            resp = requests.get(url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("Download attempt %d failed: %s", attempt + 1, e)
            time.sleep(DOWNLOAD_BACKOFF * (2 ** attempt))
    logger.error("All download attempts failed for %s", url)
    raise last_exc

def detect_content_type_from_headers_or_url(resp: requests.Response, url: str) -> str:
    """Return a normalized content-type string for decision-making."""
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"

def extract_text_from_html(html: str) -> str:
    """Extract visible text and any token-like strings from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove scripts/styles
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    visible = soup.get_text(separator="\n")
    # find long hex-like tokens or long alphanumeric tokens typical for secret tokens
    tokens = re.findall(r"[A-Fa-f0-9]{20,}|[A-Za-z0-9_\-]{20,}", visible)
    # prefer token candidates if they look like a secret
    if tokens:
        # join candidates with newline so they become searchable
        candidate_text = "\n".join(tokens)
        # also include full visible text as fallback
        return candidate_text + "\n\n" + visible
    return visible

def save_stream_to_tempfile(resp: requests.Response, suffix: str = "") -> str:
    """Save streamed response content to a temp file and return path."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in resp.iter_content(chunk_size=8192):
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

# -----------------------------
# Document loaders
# -----------------------------
def load_non_pdf(file_path: str) -> List[Document]:
    """Load a variety of non-pdf files into langchain Documents (best-effort)."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        logger.debug("Loading non-PDF file %s (ext=%s)", file_path, ext)
        if ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
            # UnstructuredFileLoader might be deprecated in some langchain versions.
            # We try to fallback gracefully.
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
    """Stream pages from a PDF and yield one Document per page (memory efficient)."""
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
    """Extract an archive to a temp dir and load contained files (best-effort)."""
    docs: List[Document] = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with archive_class(file_path) as archive:
            try:
                archive.extractall(extract_dir)
            except Exception:
                # Some libraries have different API
                try:
                    archive.extractall(path=extract_dir)
                except Exception as e:
                    logger.warning("Archive extraction fallback failed: %s", e)
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    ext = os.path.splitext(full_path)[1].lower()
                    if ext == ".pdf":
                        # leave PDFs to be processed by PDF streaming stage
                        docs.append(Document(page_content=f"[PDF IN ARCHIVE]: {full_path}", metadata={"path": full_path}))
                    else:
                        docs.extend(load_non_pdf(full_path))
    return docs

# -----------------------------
# FAISS index builder (incremental)
# -----------------------------
def build_faiss_index_from_pdf(pdf_path: str,
                               embeddings,
                               chunk_size: int = 1200,
                               chunk_overlap: int = CHUNK_OVERLAP,
                               batch_pages: int = BATCH_SIZE_PAGES,
                               max_chunks: int = MAX_CHUNKS) -> FAISS:
    """
    Build a FAISS index by streaming the PDF page-by-page and splitting batches.
    This avoids loading entire PDF into memory for huge documents.
    """
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
                if faiss_index is None:
                    logger.info("Creating initial FAISS index from %d chunks...", len(split_chunks))
                    faiss_index = FAISS.from_documents(split_chunks, embeddings)
                else:
                    logger.info("Adding %d chunks to FAISS index (total before add: %d).", len(split_chunks), total_chunks)
                    faiss_index.add_documents(split_chunks)

                total_chunks += len(split_chunks)
                logger.info("Total chunks so far: %d", total_chunks)

            batch_docs = []
            pages_in_batch = 0

            if total_chunks >= max_chunks:
                logger.info("Reached max_chunks (%d) after adding batch. Ending.", max_chunks)
                break

    # final partial batch
    if pages_in_batch > 0 and total_chunks < max_chunks:
        split_chunks = splitter.split_documents(batch_docs)
        split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        allowed = max_chunks - total_chunks
        if len(split_chunks) > allowed:
            split_chunks = split_chunks[:allowed]
        if split_chunks:
            if faiss_index is None:
                logger.info("Creating FAISS index from final batch (%d chunks)...", len(split_chunks))
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                logger.info("Adding final %d chunks to FAISS index.", len(split_chunks))
                faiss_index.add_documents(split_chunks)
            total_chunks += len(split_chunks)
            logger.info("Final total chunks: %d", total_chunks)

    if faiss_index is None:
        logger.warning("No text extracted from PDF; creating empty FAISS index.")
        faiss_index = FAISS.from_documents([Document(page_content="")], embeddings)

    return faiss_index

# -----------------------------
# Async QA helper (uses qa_chain)
# -----------------------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    top_k = 6
    docs = vector_store.similarity_search(question, k=top_k)
    if not docs:
        return "The policy document does not specify this clearly."
    raw = await chain.ainvoke({
        "context": docs,
        "input": question,
        "language": lang
    })
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# -----------------------------
# Core endpoint logic
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None), request: Request = None):
    """
    Main endpoint. Accepts JSON:
    {
      "documents": "<url>",
      "questions": ["q1", "q2", ...]
    }
    """
    global qa_chain, content_language

    logger.info("Received /hackrx/run for document: %s", data.documents)

    # Auth check if env var provided
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

        # Download resource (stream for large files)
        logger.info("Downloading document from: %s", data.documents)
        resp = _retry_request(data.documents, stream=True, timeout=PDF_STREAM_TIMEOUT)

        content_type = detect_content_type_from_headers_or_url(resp, data.documents)
        logger.info("Remote content-type detected as: %s", content_type)

        # If HTML page or text-like -> parse accordingly
        if "text/html" in content_type or ("text" in content_type and "html" not in content_type and len(resp.content) < 200000):
            # If HTML, extract meaningful token strings and visible text
            resp_text = resp.text
            extracted_text = extract_text_from_html(resp_text)
            docs = [Document(page_content=extracted_text)]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            if not chunks:
                # fallback to raw text
                chunks = [Document(page_content=resp_text)]
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("Processed HTML/text endpoint into FAISS index with %d chunks", len(chunks))

        # If JSON or plain text but not huge, treat as text doc
        elif "application/json" in content_type or "text/plain" in content_type or data.documents.lower().endswith((".json", ".txt")):
            try:
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
            except Exception as e:
                logger.exception("Failed to process JSON/text endpoint: %s", e)
                raise HTTPException(status_code=400, detail="Error reading text/JSON document.")

        else:
            # Save streamed content to temp file for file-based processing
            guessed_ext = mimetypes.guess_extension(content_type.split(";")[0]) or os.path.splitext(data.documents)[1] or ".bin"
            tmp_path = save_stream_to_tempfile(resp, suffix=guessed_ext)
            logger.info("File saved to temporary path: %s", tmp_path)

            ext = os.path.splitext(tmp_path)[1].lower()
            logger.info("Temporary file ext determined as: %s", ext)

            if ext == ".pdf":
                # For PDFs: use incremental FAISS builder by streaming pages
                try:
                    vector_store = build_faiss_index_from_pdf(
                        pdf_path=tmp_path,
                        embeddings=embeddings,
                        chunk_size=1200,
                        chunk_overlap=CHUNK_OVERLAP,
                        batch_pages=BATCH_SIZE_PAGES,
                        max_chunks=MAX_CHUNKS
                    )
                except Exception as e:
                    logger.exception("PDF indexing failed: %s", e)
                    # fallback: try extracting whole text and indexing
                    try:
                        with fitz.open(tmp_path) as pdf_doc:
                            all_text = "\n".join([p.get_text() for p in pdf_doc])
                        docs = [Document(page_content=all_text)]
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
                        chunks = splitter.split_documents(docs)
                        vector_store = FAISS.from_documents(chunks[:MAX_CHUNKS], embeddings)
                    except Exception as e2:
                        logger.exception("Full PDF extraction fallback failed: %s", e2)
                        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")

            elif ext in [".zip", ".rar", ".7z"]:
                # Extract archives and build index
                docs: List[Document] = []
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
                # Treat unknown binary as text if possible (try decode)
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

        # Language detect first stored doc if possible
        try:
            first_doc = None
            for k in vector_store.docstore._dict:
                first_doc = vector_store.docstore._dict[k]
                break
            if first_doc and first_doc.page_content:
                content_language = detect(first_doc.page_content)
            else:
                content_language = "unknown"
        except Exception:
            content_language = "unknown"

        # Answer questions concurrently using qa_chain + retrieval
        tasks = [ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        # Logging for debugging
        for q, a in zip(data.questions, answers):
            logger.info("QUESTION: %s\nANSWER: %s\n", q, a)

        total_time = time.time() - start_time
        logger.info("Processing completed in %.2f seconds.", total_time)

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        # cleanup
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
