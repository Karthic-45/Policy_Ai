import os
import tempfile
import asyncio
import requests
import zipfile
import mimetypes
import pandas as pd
import logging
import fitz  # PyMuPDF
from typing import List, Optional, Iterable
from langdetect import detect
from bs4 import BeautifulSoup

# ---------------- Force FAISS to CPU-only before importing ----------------
os.environ["FAISS_NO_GPU"] = "1"            # Don't try GPU
os.environ["FAISS_DISABLE_AVX512"] = "1"    # Disable AVX512 entirely
os.environ["FAISS_DISABLE_AVX512_SPR"] = "1"  # Disable SPR AVX512

import faiss
try:
    faiss.omp_set_num_threads(1)
    logging.info("✅ FAISS forced to CPU-only mode with AVX512 fully disabled.")
except Exception as e:
    logging.warning("⚠️ FAISS CPU initialization warning: %s", e)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ✅ Updated imports to avoid deprecation warnings
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

from PIL import Image
import rarfile
import py7zr

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.0.1"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qa_chain = None
content_language = None

# Configurable defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))

# Model initialization
try:
    logger.info("🔍 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

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
    logger.info("✅ Models initialized successfully.")
except Exception as e:
    logger.exception("❌ Error initializing models: %s", e)
    raise

# Request models
class QuestionRequest(BaseModel):
    question: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get("/")
def health():
    return {"status": "API is running"}

# ---------------- Utility loaders (non-PDF) ----------------
def load_non_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return [Document(page_content=text)]
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path).load()
        elif ext == ".eml":
            return UnstructuredEmailLoader(file_path).load()
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
                raw_data = f.read()
            try:
                decoded = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_data.decode("latin-1", errors="ignore")
            return [Document(page_content=decoded)]
    except Exception as e:
        logger.warning("Non-PDF loader failed for %s: %s", file_path, e)
        return []

# ---------------- PDF streaming (page-by-page) ----------------
def iter_pdf_pages_as_documents(pdf_path: str) -> Iterable[Document]:
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            if text.strip():
                yield Document(page_content=text.strip(), metadata={"page": pno + 1, "source": os.path.basename(pdf_path)})
    finally:
        doc.close()

# ---------------- Archive extractor ----------------
def extract_and_load(file_path, archive_class):
    docs = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with archive_class(file_path) as archive:
            try:
                archive.extractall(extract_dir)
            except Exception:
                archive.extractall(path=extract_dir)
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    if full_path.lower().endswith(".pdf"):
                        docs.append(Document(page_content=f"[PDF IN ARCHIVE]: {full_path}", metadata={"path": full_path}))
                    else:
                        docs.extend(load_non_pdf(full_path))
    return docs

# ---------------- Incremental FAISS builder ----------------
def build_faiss_index_from_pdf(pdf_path: str, embeddings, chunk_size: int, chunk_overlap: int, batch_pages: int, max_chunks: int) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    faiss_index = None
    total_chunks = 0
    batch_docs = []

    for page_doc in iter_pdf_pages_as_documents(pdf_path):
        batch_docs.append(page_doc)
        if len(batch_docs) >= batch_pages:
            split_chunks = [c for c in splitter.split_documents(batch_docs) if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            allowed = max_chunks - total_chunks
            if allowed <= 0:
                break
            split_chunks = split_chunks[:allowed]
            if split_chunks:
                if faiss_index is None:
                    faiss_index = FAISS.from_documents(split_chunks, embeddings)
                else:
                    faiss_index.add_documents(split_chunks)
                total_chunks += len(split_chunks)
            batch_docs = []

    if batch_docs and total_chunks < max_chunks:
        split_chunks = [c for c in splitter.split_documents(batch_docs) if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        split_chunks = split_chunks[:max_chunks - total_chunks]
        if faiss_index is None:
            faiss_index = FAISS.from_documents(split_chunks, embeddings)
        else:
            faiss_index.add_documents(split_chunks)

    return faiss_index or FAISS.from_documents([Document(page_content="")], embeddings)

# ---------------- Async QA helper ----------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    docs = vector_store.similarity_search(question, k=6)
    if not docs:
        return "The policy document does not specify this clearly."
    raw = await chain.ainvoke({"context": docs, "input": question, "language": lang})
    answer = raw.strip()
    return answer if answer and "i don't know" not in answer.lower() else "The policy document does not specify this clearly."

# ---------------- Main /hackrx/run endpoint ----------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    global content_language
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    logger.info("📥 /hackrx/run request received for document: %s", data.documents)

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if authorization.split("Bearer ")[1] != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    tmp_path = None
    try:
        start_time = asyncio.get_event_loop().time()
        resp = requests.get(data.documents, stream=True, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")

        content_type = resp.headers.get("content-type", "").split(";")[0]
        if "text/html" in content_type.lower():
            logger.info("📄 HTML document detected. Processing as HTML file.")
            extension = ".html"
        else:
            extension = mimetypes.guess_extension(content_type) or os.path.splitext(data.documents)[1] or ".bin"

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tf:
            for chunk in resp.iter_content(chunk_size=8192):
                tf.write(chunk)
            tmp_path = tf.name

        ext = os.path.splitext(tmp_path)[1].lower()
        vector_store = None

        if ext == ".pdf":
            try:
                page_count = fitz.open(tmp_path).page_count
            except Exception:
                page_count = 0
            chunk_size = 600 if page_count <= 10 else 1000 if page_count <= 200 else 1200 if page_count <= 800 else 1500
            vector_store = build_faiss_index_from_pdf(tmp_path, embeddings, chunk_size, CHUNK_OVERLAP, BATCH_SIZE_PAGES, MAX_CHUNKS)
        else:
            docs = []
            if ext in [".zip", ".rar", ".7z"]:
                if ext == ".zip":
                    docs = extract_and_load(tmp_path, zipfile.ZipFile)
                elif ext == ".rar":
                    docs = extract_and_load(tmp_path, rarfile.RarFile)
                else:
                    docs = extract_and_load(tmp_path, py7zr.SevenZipFile)
            else:
                docs = load_non_pdf(tmp_path)
            docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= MIN_CHUNK_LEN]
            if not docs:
                raise HTTPException(status_code=400, detail="No readable content found in document.")
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)[:MAX_CHUNKS]
            vector_store = FAISS.from_documents(chunks, embeddings)

        try:
            first_doc = next(iter(vector_store.docstore._dict.values()), None)
            content_language = detect(first_doc.page_content) if first_doc and first_doc.page_content else "unknown"
        except Exception:
            content_language = "unknown"

        answers = await asyncio.gather(*[ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions])
        logger.info("✅ Processing complete in %.2f seconds.", asyncio.get_event_loop().time() - start_time)
        return {"answers": answers}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
