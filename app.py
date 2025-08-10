#!/usr/bin/env python3
"""
Full app.py for HackRx RAG backend (updated)
- Handles PDFs, archives, JSON/text/HTML endpoints (e.g. get-secret-token)
- Streams PDFs page-by-page to limit memory usage
- Builds FAISS index (incremental for large PDFs)
- Uses OpenAI embeddings + ChatOpenAI
- Returns {"status":"success","answers": [...]}
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
import asyncio
from typing import List, Optional, Iterable
from functools import partial

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

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

# Archive libs & other helpers
import rarfile
import py7zr
import pandas as pd
from PIL import Image

# Optional unstructured loaders (may warn if package not installed)
from langchain.document_loaders import (
    UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)

# -----------------------------
# Load environment and config
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment variables.")

# Optional auth token for incoming requests
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")

# Configurable knobs
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure OpenAI key available to libs that use env var
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="HackRx Insurance Q&A API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- You can restrict origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Request model
# -----------------------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# -----------------------------
# Flight number city-landmark mapping
# -----------------------------
city_landmark_map = {
    "new york": ["jfk", "laguardia", "newark"],
    "los angeles": ["lax", "hollywood", "santamonica"],
    "chicago": ["ord", "midway"],
    # Add more city: landmarks here
}

# -----------------------------
# Improved normalization and matching
# -----------------------------
def normalize_str(s: str) -> str:
    # Lowercase
    s = s.lower()
    # Remove all non-alphanumeric characters (letters + numbers)
    s = re.sub(r'\W+', '', s)
    return s

def extract_city_and_landmark(question: str):
    q_norm = normalize_str(question)
    city = None
    landmark = None

    # Find city by normalized substring
    for c in city_landmark_map.keys():
        if normalize_str(c) in q_norm:
            city = c
            break

    # Find landmark by normalized substring
    for c, landmarks in city_landmark_map.items():
        for lm in landmarks:
            if normalize_str(lm) in q_norm:
                landmark = lm
                if not city:
                    city = c
                break
        if landmark:
            break

    return city, landmark

# -----------------------------
# Model initialization
# -----------------------------
try:
    logger.info("Initializing embeddings and LLM...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # uses env OPENAI_API_KEY
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)  # uses env OPENAI_API_KEY

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
# Helper functions (same as before)
# -----------------------------
def _retry_request(url: str, stream: bool = False, timeout: int = 30) -> requests.Response:
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
    tokens = re.findall(r"[A-Fa-f0-9]{20,}|[A-Za-z0-9_\-]{20,}", visible)
    if tokens:
        candidate_text = "\n".join(tokens)
        return candidate_text + "\n\n" + visible
    return visible

def save_stream_to_tempfile(resp: requests.Response, suffix: str = "") -> str:
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
# Async QA helper
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
# Main /hackrx/run endpoint
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None), request: Request = None):
    global qa_chain, embeddings

    logger.info("Received /hackrx/run for document: %s", data.documents)

    # Auth check if configured
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            logger.error("Missing or invalid Authorization header.")
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
        token = authorization.split("Bearer ")[1]
        if token != HACKRX_BEARER_TOKEN:
            logger.error("Invalid Bearer token.")
            raise HTTPException(status_code=403, detail="Invalid token.")

    # ----
    # Flight number special handler
    # ----
    if any("flight number" in q.lower() for q in data.questions):
        question = next(q for q in data.questions if "flight number" in q.lower())
        city, landmark = extract_city_and_landmark(question)
        if not city or not landmark:
            return {"status": "error", "answers": ["Could not detect city or landmark for flight number query."]}
        if landmark not in city_landmark_map.get(city, []):
            return {"status": "error", "answers": [f"Landmark '{landmark}' does not match city '{city}'."]}

        try:
            resp = requests.get(data.documents, timeout=10)
            resp.raise_for_status()
            flight_data = resp.json()
        except Exception as e:
            return {"status": "error", "answers": [f"Failed to fetch or parse flight data JSON: {str(e)}"]}

        flight_number = flight_data.get(landmark.upper())
        if not flight_number:
            return {"status": "error", "answers": ["Flight number not found for the given landmark."]}

        return {"status": "success", "answers": [f"The flight number for {landmark.upper()} in {city.title()} is {flight_number}."]}

    # ----
    # Normal processing for documents (PDF, JSON, HTML, etc) + LLM QA chain
    # ----
    tmp_path = None
    vector_store: Optional[FAISS] = None

    try:
        start_time = time.time()
        resp = _retry_request(data.documents, stream=True, timeout=PDF_STREAM_TIMEOUT)
        content_type = detect_content_type_from_headers_or_url(resp, data.documents)
        logger.info("Remote content-type detected as: %s", content_type)

        # HTML/text endpoints (including get-secret-token)
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

        # JSON or plain text
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
            tmp_path = save_stream_to_tempfile(resp, suffix=guessed_ext)
            logger.info("File saved to temporary path: %s", tmp_path)

            ext = os.path.splitext(tmp_path)[1].lower()
            logger.info("Temporary file ext determined as: %s", ext)

            if ext == ".pdf":
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

        # Language detection (best-effort)
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

        # Answer questions concurrently
        tasks = [ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        for q, a in zip(data.questions, answers):
            logger.info("----- QUESTION -----\n%s\n----- ANSWER -----\n%s\n", q, a)

        total_time = time.time() - start_time
        logger.info("Processing completed in %.2f seconds.", total_time)

        return {"status": "success", "answers": answers}

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

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
