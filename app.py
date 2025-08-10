#!/usr/bin/env python3
"""
Full app.py for HackRx RAG backend (updated)
- Keeps your original PDF / archive / HTML / JSON handling
- Streams PDFs page-by-page to limit memory usage
- Builds FAISS index (incremental for large PDFs)
- Uses OpenAI embeddings + ChatOpenAI
- Behavior for flight-number questions:
    1) Ask the QA chain (GPT) first.
    2) If GPT returns a plausible flight number, use it.
    3) Otherwise automatically follow the PDF's flow:
       - call /submissions/myFavouriteCity to get the city
       - map city -> landmark by extracting mapping from the PDF text
       - call the correct flights endpoint per the PDF instructions
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
from typing import List, Optional, Iterable, Tuple, Dict
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
app = FastAPI(title="HackRx Insurance Q&A API", version="1.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request model
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
# Helper functions
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
    """
    Extract visible text and token-like strings from HTML.
    Returns token candidates first (if any) followed by visible text.
    """
    soup = BeautifulSoup(html, "html.parser")
    for el in soup(["script", "style", "noscript"]):
        el.decompose()
    visible = soup.get_text(separator="\n")
    # find long hex-like tokens or long alphanumeric tokens typical for secret tokens
    tokens = re.findall(r"[A-Fa-f0-9]{16,}|[A-Za-z0-9_\-]{16,}", visible)
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
# Document loaders & extractors
# -----------------------------
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

# -----------------------------
# FAISS builder (incremental)
# -----------------------------
def build_faiss_index_from_pdf(pdf_path: str,
                               embeddings,
                               chunk_size: int = 1200,
                               chunk_overlap: int = CHUNK_OVERLAP,
                               batch_pages: int = BATCH_SIZE_PAGES,
                               max_chunks: int = MAX_CHUNKS) -> FAISS:
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
# Helper: find token-like strings inside docs
# -----------------------------
def find_token_in_docs(docs: List[Document]) -> Optional[str]:
    """
    Search documents for token-like strings (hex or long alphanumerics).
    Return the first match or None.
    """
    token_pattern = re.compile(r"[A-Fa-f0-9]{16,}|[A-Za-z0-9_\-]{16,}")
    for d in docs:
        if not d or not d.page_content:
            continue
        for m in token_pattern.findall(d.page_content):
            # filter out very common words by requiring mixed characters or digits count
            if len(m) >= 16:
                return m
    return None

# -----------------------------
# New helper: extract city->landmark mapping from raw PDF text (dynamic, not hardcoded)
# -----------------------------
def extract_city_landmark_map_from_text(raw_text: str) -> Dict[str, str]:
    """
    Parse the PDF plain text and attempt to extract lines that look like:
      <emoji?> LandmarkName CityName
    or split-on-multiple-spaces lines with landmark and city.
    Returns dict: city.lower() -> landmark (original case)
    This is heuristic but tuned for the table format in the sample.
    """
    mapping: Dict[str, str] = {}
    # Normalize some invisible whitespace
    cleaned = raw_text.replace('\u00a0', ' ')
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    # Heuristics: look for lines with landmark then city (last token(s) look like a city)
    for ln in lines:
        # skip header-like lines
        if len(ln) < 4:
            continue
        # remove leading bullets/emojis if present
        ln_clean = re.sub(r'^[\W_]+', '', ln).strip()
        # If there are multiple big spaces or tab, split
        if re.search(r'\s{2,}', ln_clean):
            parts = re.split(r'\s{2,}', ln_clean)
            if len(parts) >= 2:
                landmark = parts[0].strip()
                city = parts[-1].strip()
                if landmark and city and len(city) < 40:
                    mapping[city.lower()] = landmark
                    continue
        # Otherwise try rsplit once: last word(s) likely the city
        parts = ln_clean.rsplit(' ', 1)
        if len(parts) == 2:
            maybe_landmark, maybe_city = parts[0].strip(), parts[1].strip()
            # city should start with uppercase letter
            if maybe_city and maybe_city[0].isupper() and len(maybe_city) <= 30 and len(maybe_landmark) > 0:
                mapping[maybe_city.lower()] = maybe_landmark
                continue
        # As a fallback, try rsplit two tokens
        parts2 = ln_clean.rsplit(' ', 2)
        if len(parts2) == 3:
            maybe_landmark = parts2[0].strip()
            maybe_city = (parts2[1] + " " + parts2[2]).strip()
            if maybe_city and maybe_city[0].isupper() and len(maybe_city) <= 40:
                mapping[maybe_city.lower()] = maybe_landmark
                continue
    return mapping

# -----------------------------
# Helper: choose flight endpoint for a landmark (these are the rules in the PDF)
# -----------------------------
def choose_flight_endpoint_for_landmark(landmark: str) -> str:
    """
    Based on the PDF instructions:
      - Gateway of India -> getFirstCityFlightNumber
      - Taj Mahal -> getSecondCityFlightNumber
      - Eiffel Tower -> getThirdCityFlightNumber
      - Big Ben -> getFourthCityFlightNumber
      - otherwise -> getFifthCityFlightNumber
    """
    base = "https://register.hackrx.in/teams/public/flights/"
    if not landmark:
        return base + "getFifthCityFlightNumber"
    key = landmark.strip().lower()
    # Compound logical checks grouped to avoid precedence surprises
    if ("gateway of india" in key) or ("gateway" in key and "india" in key):
        return base + "getFirstCityFlightNumber"
    if "taj mahal" in key or "taj" in key:
        return base + "getSecondCityFlightNumber"
    if "eiffel" in key or "eiffel tower" in key:
        return base + "getThirdCityFlightNumber"
    if "big ben" in key or ("big" in key and "ben" in key):
        return base + "getFourthCityFlightNumber"
    return base + "getFifthCityFlightNumber"

# -----------------------------
# Helper: read entire PDF text (safe fallback) and return mapping
# -----------------------------
def read_pdf_full_text(pdf_path: str) -> str:
    try:
        with fitz.open(pdf_path) as pdf_doc:
            pages = []
            for p in pdf_doc:
                pages.append(p.get_text("text") or "")
            return "\n".join(pages)
    except Exception as e:
        logger.warning("Failed to read full PDF text: %s", e)
        return ""

# -----------------------------
# Async QA helper (with token shortcut)
# -----------------------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    """
    If question asks specifically for a 'secret token' or 'token', try to find it
    directly from the indexed documents before calling the LLM.
    Otherwise, run RAG + chain.
    """
    q_lower = question.lower()
    # If question explicitly asks to "get the secret token" or "return the token", search docs.
    if any(phrase in q_lower for phrase in ["secret token", "get the token", "return the token", "secret-token", "get-secret"]):
        # grab stored docs from vector_store
        docs_list = []
        try:
            for v in vector_store.docstore._dict.values():
                docs_list.append(v)
        except Exception:
            # fallback to similarity search across a simple query to pull some docs
            try:
                docs_list = vector_store.similarity_search("token", k=10)
            except Exception:
                docs_list = []

        found = find_token_in_docs(docs_list)
        if found:
            return found

    # Fall back to normal retrieval + chain
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

    # chain.ainvoke may return a dict or string depending on chain implementation
    if isinstance(raw, dict):
        # common fields: 'text', 'answer', 'output_text'
        for key in ("output_text", "answer", "text", "response", "content"):
            if key in raw and isinstance(raw[key], str) and raw[key].strip():
                answer = raw[key].strip()
                break
        else:
            # fallback to json dump
            answer = json.dumps(raw, ensure_ascii=False)
    elif isinstance(raw, str):
        answer = raw.strip()
    else:
        answer = str(raw).strip()

    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# -----------------------------
# Utility: validate whether a GPT answer looks like a plausible flight number
# -----------------------------
def is_plausible_flight_number(s: str) -> Optional[str]:
    """
    Heuristic: return a plausible token if present in s.
    Accepts simple alphanumeric sequences of length >=2, commonly flight numbers are alnum.
    If no plausible token, return None.
    """
    if not s:
        return None
    # common patterns: letters+digits (e.g., "AA123"), or single alphanumeric token
    # We'll look for the first token with letters or digits of length >=2
    m = re.search(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{1,}\b", s)
    if m:
        token = m.group(0).strip()
        # additional sanity: avoid returning generic words like "flight" or "number"
        if re.search(r"flight|number|the|is|please", token, re.IGNORECASE):
            # try to search for the next candidate
            for mm in re.finditer(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{1,}\b", s):
                tok = mm.group(0).strip()
                if not re.search(r"flight|number|the|is|please", tok, re.IGNORECASE):
                    return tok
            return None
        return token
    return None

# -----------------------------
# Main /hackrx/run endpoint (updated: GPT-first for flight-number Qs)
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

    tmp_path = None
    vector_store: Optional[FAISS] = None
    pdf_full_text = ""

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
                # also capture full PDF text for dynamic mapping extraction
                try:
                    pdf_full_text = read_pdf_full_text(tmp_path)
                except Exception:
                    pdf_full_text = ""

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

        # ---------------------------------------------------------------------
        # Special-case: flight-number questions: GPT-first, then auto-fetch if GPT fails
        # ---------------------------------------------------------------------
        answers = [None] * len(data.questions)

        # Identify flight-related question indices
        flight_question_indices: List[int] = []
        non_flight_question_indices: List[int] = []
        for idx, q in enumerate(data.questions):
            if re.search(r"\bflight\s*number\b|\bflight\b|\bflight-number\b", q, re.IGNORECASE):
                flight_question_indices.append(idx)
            else:
                non_flight_question_indices.append(idx)

        # 1) Ask GPT for all questions (we need GPT answers for flight-questions first)
        # Prepare tasks for GPT for flight questions (and optionally other questions together)
        # We'll ask GPT for flight questions first, to obey "letting GPT try first"
        gpt_tasks = []
        gpt_indices = []
        for idx in flight_question_indices:
            gpt_tasks.append(ask_async_chain(qa_chain, vector_store, data.questions[idx].strip()))
            gpt_indices.append(idx)

        # Run GPT for flight questions
        gpt_results = []
        if gpt_tasks:
            gpt_results = await asyncio.gather(*gpt_tasks)

        # Evaluate GPT results for plausibility
        flight_answers: Dict[int, str] = {}
        unresolved_flight_indices: List[int] = []
        for idx, res in zip(gpt_indices, gpt_results):
            res_text = None
            if isinstance(res, dict):
                for key in ("output_text", "answer", "text", "response", "content"):
                    if key in res and isinstance(res[key], str) and res[key].strip():
                        res_text = res[key].strip()
                        break
                if res_text is None:
                    res_text = json.dumps(res, ensure_ascii=False)
            else:
                res_text = str(res).strip()

            plausible = is_plausible_flight_number(res_text)
            if plausible:
                flight_answers[idx] = f"Flight number: {plausible}"
                logger.info("GPT-provided plausible flight for question %d: %s", idx, plausible)
            else:
                # keep GPT response as initial (for transparency), but mark unresolved to auto-fetch
                logger.info("GPT response not a plausible flight (question %d): %s", idx, res_text)
                unresolved_flight_indices.append(idx)

        # 2) For unresolved flight questions, perform automatic flow using PDF instructions
        if unresolved_flight_indices:
            try:
                # 2a) Get favourite city
                fav_city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
                logger.info("Querying favourite city endpoint: %s", fav_city_url)
                fav_resp = _retry_request(fav_city_url, stream=False, timeout=15)
                fav_resp.raise_for_status()
                try:
                    fav_json = fav_resp.json()
                    fav_city = None
                    if isinstance(fav_json, dict):
                        for key in ("city", "favouriteCity", "favoriteCity", "myFavouriteCity", "myFavoriteCity", "data"):
                            if key in fav_json and isinstance(fav_json[key], str) and fav_json[key].strip():
                                fav_city = fav_json[key].strip()
                                break
                        if fav_city is None:
                            # find any string value
                            for v in fav_json.values():
                                if isinstance(v, str) and v.strip():
                                    fav_city = v.strip()
                                    break
                    else:
                        fav_city = str(fav_json).strip()
                except Exception:
                    fav_city = fav_resp.text.strip()

                fav_city_simple = (fav_city or "").strip().strip('"').strip("'")
                logger.info("Favourite city from API: %s", fav_city_simple)

                # 2b) Build mapping city -> landmark from PDF full text first
                city_landmark_map: Dict[str, str] = {}
                if pdf_full_text:
                    city_landmark_map = extract_city_landmark_map_from_text(pdf_full_text)
                    logger.info("Extracted %d city->landmark entries from PDF pre-scan", len(city_landmark_map))

                # 2c) If mapping is missing or doesn't include the favorite, try to extract from vector_store docs
                if (not city_landmark_map) or (fav_city_simple and fav_city_simple.lower() not in city_landmark_map):
                    try:
                        vs_text_parts = []
                        for v in vector_store.docstore._dict.values():
                            if hasattr(v, "page_content") and v.page_content:
                                vs_text_parts.append(v.page_content)
                            elif isinstance(v, dict):
                                vs_text_parts.append(json.dumps(v))
                        vs_text = "\n".join(vs_text_parts)
                        if vs_text:
                            additional_map = extract_city_landmark_map_from_text(vs_text)
                            for k, v in additional_map.items():
                                if k not in city_landmark_map:
                                    city_landmark_map[k] = v
                            logger.info("Merged %d additional mapping entries from vector_store", len(additional_map))
                    except Exception as e:
                        logger.debug("Could not construct mapping from vector_store: %s", e)

                # 2d) Determine landmark for favourite city
                landmark = None
                if fav_city_simple:
                    landmark = city_landmark_map.get(fav_city_simple.lower())
                if not landmark and fav_city_simple:
                    lowered = fav_city_simple.lower()
                    for city_key, lm in city_landmark_map.items():
                        if lowered and lowered in city_key:
                            landmark = lm
                            break
                logger.info("Pre-scan found city=%s landmark=%s", fav_city_simple or None, landmark or None)

                # 2e) Choose endpoint and call it
                flight_endpoint_url = choose_flight_endpoint_for_landmark(landmark or "")
                logger.info("Calling flight endpoint: %s", flight_endpoint_url)
                flight_number_result = None
                try:
                    flight_resp = _retry_request(flight_endpoint_url, stream=False, timeout=15)
                    flight_resp.raise_for_status()
                    # parse JSON if possible
                    try:
                        frj = flight_resp.json()
                        if isinstance(frj, dict):
                            for val in frj.values():
                                if isinstance(val, (str, int)) and str(val).strip():
                                    flight_number_result = str(val).strip()
                                    break
                            if not flight_number_result:
                                flight_number_result = json.dumps(frj, ensure_ascii=False).strip()
                        else:
                            flight_number_result = str(frj).strip()
                    except Exception:
                        flight_text = flight_resp.text.strip()
                        flight_number_result = flight_text
                except Exception as e:
                    logger.warning("Flight endpoint call failed: %s", e)
                    flight_number_result = None

                final_flight = None
                if flight_number_result:
                    m = re.search(r"[A-Za-z0-9_\-]{2,}", flight_number_result)
                    if m:
                        final_flight = m.group(0)
                    else:
                        final_flight = flight_number_result.strip()

                for idx in unresolved_flight_indices:
                    if final_flight:
                        flight_answers[idx] = f"Flight number: {final_flight}"
                    else:
                        flight_answers[idx] = "Could not retrieve flight number automatically."

            except Exception as e:
                logger.exception("Automatic flight-number flow failed: %s", e)
                # If everything fails, leave unresolved to be answered by RAG below

        # Place flight answers into final answers array
        for idx, ans in flight_answers.items():
            answers[idx] = ans

        # ---------------------------------------------------------------------
        # For all remaining unanswered questions, use the RAG chain (GPT)
        # ---------------------------------------------------------------------
        pending_tasks = []
        pending_indices = []
        for idx, q in enumerate(data.questions):
            if answers[idx] is not None:
                continue
            pending_indices.append(idx)
            pending_tasks.append(ask_async_chain(qa_chain, vector_store, q.strip()))

        if pending_tasks:
            pending_results = await asyncio.gather(*pending_tasks)
            for idx, res in zip(pending_indices, pending_results):
                if isinstance(res, dict):
                    s = res.get("output_text") or res.get("answer") or res.get("text") or json.dumps(res, ensure_ascii=False)
                    answers[idx] = str(s).strip()
                else:
                    answers[idx] = str(res).strip()

        # Final normalization
        normalized_answers = [ (a if isinstance(a, str) else str(a)) for a in answers ]

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

# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
