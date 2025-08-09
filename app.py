import os
import tempfile
import asyncio
import requests
import zipfile
import mimetypes
import pandas as pd
import logging
import fitz  # PyMuPDF
import math
from typing import List, Optional, Iterable
from langdetect import detect

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Optional loaders used for non-pdf types
from langchain.document_loaders import (
    UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)
from PIL import Image
import rarfile
import py7zr

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ------------------------------------------------

# Load env
load_dotenv()

# FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API (Large PDF friendly)",
    description="Answer insurance-related questions using RAG + GPT. Optimized for very large PDFs.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Configurable Constants ----------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))  # pages to read before creating/supplementing index
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))  # absolute cap on chunks to avoid explosion
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))  # ignore very tiny chunks
# -------------------------------------------------------

# Globals
qa_chain = None
content_language = None

# Initialize models
try:
    logger.info("Initializing OpenAI models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in environment variables.")
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
    logger.info("Models initialized.")
except Exception as e:
    logger.exception("Error initializing models: %s", e)
    raise

# Request schemas
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health
@app.get("/")
def health():
    return {"status": "API is running"}

# ---------------- Utility: Non-PDF loaders (kept simple) ----------------
def load_non_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
            return UnstructuredFileLoader(file_path).load()
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
                return [Document(page_content=f"[BINARY XLSX PREVIEW] {raw[:512].hex()}")]
        elif ext in [".zip", ".rar", ".7z"]:
            # fallback: rely on archive extractor used elsewhere; return empty here
            return []
        else:
            with open(file_path, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = raw.decode("latin-1", errors="ignore")
            return [Document(page_content=text)]
    except Exception as e:
        logger.warning("Non-PDF file load failed for %s: %s", file_path, e)
        return []

# ---------------- PDF streaming loader (page-by-page) ----------------
def iter_pdf_pages_as_documents(pdf_path: str) -> Iterable[Document]:
    """
    Stream pages from a PDF file using PyMuPDF (fitz).
    Returns Document objects, one per page (with simple metadata).
    This avoids loading all pages' text into memory at once.
    """
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            # small normalization
            text = text.strip()
            if not text:
                # try extracting blocks if text empty (scanned pages won't work here without OCR)
                continue
            metadata = {"page": pno + 1, "source": os.path.basename(pdf_path)}
            yield Document(page_content=text, metadata=metadata)
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
                # py7zr uses extractall differently
                archive.extractall(path=extract_dir)
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    ext = os.path.splitext(full_path)[1].lower()
                    if ext == ".pdf":
                        # defer to pdf streaming during indexing stage
                        docs.append(Document(page_content=f"[PDF IN ARCHIVE]: {full_path}", metadata={"path": full_path}))
                    else:
                        docs.extend(load_non_pdf(full_path))
    return docs

# ---------------- Chunking + incremental FAISS building ----------------
def build_faiss_index_from_pdf(pdf_path: str,
                               embeddings,
                               chunk_size: int = 1200,
                               chunk_overlap: int = CHUNK_OVERLAP,
                               batch_pages: int = BATCH_SIZE_PAGES,
                               max_chunks: int = MAX_CHUNKS) -> FAISS:
    """
    Stream PDF pages, split into chunks, and create FAISS index incrementally in batches.
    Returns a FAISS vectorstore instance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    faiss_index = None
    total_chunks = 0
    page_iter = iter_pdf_pages_as_documents(pdf_path)

    batch_docs = []
    pages_in_batch = 0

    for page_doc in page_iter:
        pages_in_batch += 1
        batch_docs.append(page_doc)

        # Once we've collected enough pages, split them and add to index
        if pages_in_batch >= batch_pages:
            split_chunks = splitter.split_documents(batch_docs)
            # Filter very small chunks
            split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            if not split_chunks:
                batch_docs = []
                pages_in_batch = 0
                continue

            # Enforce remaining budget
            allowed = max_chunks - total_chunks
            if allowed <= 0:
                logger.info("Reached max_chunks cap (%d). Stopping indexing.", max_chunks)
                break
            if len(split_chunks) > allowed:
                split_chunks = split_chunks[:allowed]

            if faiss_index is None:
                logger.info("Creating initial FAISS index from first batch of %d chunks...", len(split_chunks))
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                logger.info("Adding %d chunks to FAISS index (total before add: %d).", len(split_chunks), total_chunks)
                faiss_index.add_documents(split_chunks)

            total_chunks += len(split_chunks)
            logger.info("Total chunks so far: %d", total_chunks)

            # reset batch buffers
            batch_docs = []
            pages_in_batch = 0

            # If we've hit the cap, stop early
            if total_chunks >= max_chunks:
                logger.info("Reached or exceeded max_chunks (%d). Ending indexing.", max_chunks)
                break

    # After loop, handle any remaining pages in final partial batch
    if pages_in_batch > 0 and total_chunks < max_chunks:
        split_chunks = splitter.split_documents(batch_docs)
        split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        allowed = max_chunks - total_chunks
        if len(split_chunks) > allowed:
            split_chunks = split_chunks[:allowed]
        if split_chunks:
            if faiss_index is None:
                logger.info("Creating FAISS index from final batch (chunks: %d)...", len(split_chunks))
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                logger.info("Adding final %d chunks to FAISS index.", len(split_chunks))
                faiss_index.add_documents(split_chunks)
            total_chunks += len(split_chunks)
            logger.info("Final total chunks: %d", total_chunks)

    if faiss_index is None:
        # No textual pages were found (empty PDF). Return empty FAISS built from an empty doc list
        logger.warning("No text extracted from PDF; creating empty FAISS index.")
        faiss_index = FAISS.from_documents([Document(page_content="")], embeddings)

    return faiss_index

# ---------------- Async QA helper ----------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    # Pull top-k quickly (k tuned small for speed)
    top_k = 6
    docs = vector_store.similarity_search(question, k=top_k)
    # If zero docs, reply with fallback
    if not docs:
        return "The policy document does not specify this clearly."

    # invoke chain asynchronously
    raw = await chain.ainvoke({
        "context": docs,
        "input": question,
        "language": lang
    })
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# ---------------- Main /hackrx/run endpoint ----------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    logger.info("/hackrx/run called for document: %s", data.documents)

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    # Download document to temp file
    try:
        resp = requests.get(data.documents, timeout=30)
    except Exception as e:
        logger.error("Failed to download document: %s", e)
        raise HTTPException(status_code=400, detail="Failed to download document.")

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download document. HTTP {resp.status_code}")

    content_type = resp.headers.get("content-type", "")
    extension = mimetypes.guess_extension(content_type.split(";")[0]) or os.path.splitext(data.documents)[1] or ".bin"

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmpf:
        tmpf.write(resp.content)
        tmp_path = tmpf.name

    logger.info("Saved temp file to %s", tmp_path)

    try:
        # If it's a PDF, stream-build FAISS index
        ext = os.path.splitext(tmp_path)[1].lower()
        vector_store = None

        if ext == ".pdf":
            # Choose chunk size based on approx page count (we can try to read page count quickly)
            try:
                pdf_doc = fitz.open(tmp_path)
                page_count = pdf_doc.page_count
                pdf_doc.close()
            except Exception:
                page_count = 0

            # dynamic chunk_size strategy
            if page_count == 0:
                chunk_size = 1000
            else:
                # larger docs -> larger chunks (fewer embeddings)
                if page_count <= 10:
                    chunk_size = 600
                elif page_count <= 200:
                    chunk_size = 1000
                elif page_count <= 800:
                    chunk_size = 1200
                else:
                    chunk_size = 1500

            logger.info("PDF detected. page_count=%d chunk_size=%d", page_count, chunk_size)

            # Build FAISS index streaming
            vector_store = build_faiss_index_from_pdf(
                pdf_path=tmp_path,
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=CHUNK_OVERLAP,
                batch_pages=BATCH_SIZE_PAGES,
                max_chunks=MAX_CHUNKS
            )

        else:
            # Non-pdf: use existing loader (kept simple) -> load all since these are usually small
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
            # split and create index (small files)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = chunks[:MAX_CHUNKS]
            vector_store = FAISS.from_documents(chunks, embeddings)

        # detect document language (try first chunk)
        try:
            first_text = vector_store.docstore._dict[next(iter(vector_store.docstore._dict))].page_content
            content_lang = detect(first_text)
        except Exception:
            content_lang = "unknown"

        # Concurrently answer questions
        tasks = [ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        # Final result: return only answers array
        return {"answers": answers}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Unexpected error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
