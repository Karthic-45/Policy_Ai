#!/usr/bin/env python3
"""
HackRx Flight Number API — fixed, crash-resistant single-file implementation
Accepts JSON body with `documents` (PDF URL) and `questions` list.
"""

import os
import tempfile
import logging
import mimetypes
import time
import re
from typing import Optional, Iterable, Dict, Tuple, List, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to import optional heavy deps and keep helpful flags
_HAS_REQUESTS = True
_HAS_FITZ = True
_HAS_LANGCHAIN = True
try:
    import requests
except Exception:
    _HAS_REQUESTS = False

try:
    import fitz  # PyMuPDF
except Exception:
    _HAS_FITZ = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except Exception:
    _HAS_LANGCHAIN = False

# -----------------------------
# Logging (fixed)
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")

# Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "2"))

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="HackRx Flight Number API", version="1.0.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Helpers
# -----------------------------
CLEAN_RE = re.compile(r"[^A-Za-z0-9\s]")

class DependencyError(Exception):
    pass

def _ensure_deps_for_pdf_and_requests():
    if not _HAS_REQUESTS:
        raise DependencyError("Missing dependency: requests")
    if not _HAS_FITZ:
        raise DependencyError("Missing dependency: PyMuPDF (fitz)")
    if not _HAS_LANGCHAIN:
        raise DependencyError("Missing dependency: langchain")

def _retry_request(url: str, stream=False, timeout: float = REQUEST_TIMEOUT, retries: int = DOWNLOAD_RETRIES) -> requests.Response:
    if not _HAS_REQUESTS:
        raise DependencyError("requests not available")
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("Request attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.5 * (2 ** attempt))
    raise last_exc

def detect_content_type(resp: requests.Response, url: str) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"

class TemporaryFile:
    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        self.name = None
    def __enter__(self):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=self.suffix)
        self.name = tf.name
        tf.close()
        return self.name
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.name and os.path.exists(self.name):
                os.remove(self.name)
        except Exception:
            logger.warning("Failed to cleanup temp file: %s", self.name)

def save_stream(resp: requests.Response, suffix: str = "") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        for chunk in resp.iter_content(8192):
            if chunk:
                tf.write(chunk)
        tf.flush()
        return tf.name
    finally:
        tf.close()

def iter_pdf_pages(pdf_path: str) -> Iterable[Any]:
    if not _HAS_FITZ:
        raise DependencyError("PyMuPDF not available")
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            text = doc.load_page(pno).get_text("text") or ""
            text = text.strip()
            if text:
                if _HAS_LANGCHAIN:
                    yield Document(page_content=text, metadata={"page": pno + 1})
                else:
                    yield {"page_content": text, "metadata": {"page": pno + 1}}
    finally:
        doc.close()

def build_faiss_from_pdf(pdf_path: str) -> Tuple[Any, List[Any]]:
    if not _HAS_LANGCHAIN or not _HAS_FITZ or not _HAS_REQUESTS:
        raise DependencyError("Missing required libraries for vector store")

    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        raise DependencyError("Missing FAISS vector store")

    try:
        from langchain_openai import OpenAIEmbeddings
    except Exception:
        from langchain.embeddings import OpenAIEmbeddings

    if not OPENAI_API_KEY:
        raise DependencyError("OPENAI_API_KEY not set")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=CHUNK_OVERLAP)
    docs = list(iter_pdf_pages(pdf_path))
    if not docs:
        raise ValueError("No text extracted from PDF")
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len((c.page_content if hasattr(c, "page_content") else c["page_content"]).strip()) >= MIN_CHUNK_LEN]
    if not chunks:
        raise ValueError("No valid chunks after splitting PDF")
    vs = FAISS.from_documents(chunks, embeddings)
    return vs, chunks

# -----------------------------
# Flight resolution logic
# -----------------------------
def _normalize_city_key(s: str) -> str:
    s = (s or "").strip()
    s = CLEAN_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def extract_city_landmark_map(docs) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cleaned = CLEAN_RE.sub("", line).strip()
            parts = cleaned.split()
            if len(parts) < 2:
                continue
            city_candidate = parts[-1]
            landmark_candidate = " ".join(parts[:-1]).strip()
            city_key = _normalize_city_key(city_candidate)
            if not city_key:
                continue
            mapping[city_key] = landmark_candidate
    return mapping

def _get_favourite_city() -> Optional[str]:
    if not _HAS_REQUESTS:
        return None
    try:
        resp = _retry_request("https://register.hackrx.in/submissions/myFavouriteCity", timeout=REQUEST_TIMEOUT)
        try:
            j = resp.json()
            if isinstance(j, str):
                city = j
            elif isinstance(j, dict):
                city = j.get("city") or j.get("favouriteCity") or next(iter(j.values()), None)
            else:
                city = None
        except Exception:
            city = resp.text.strip()
        if not city:
            return None
        return CLEAN_RE.sub("", str(city)).strip()
    except Exception:
        return None

def resolve_flight_number_from_docs(vector_store: Any, source_docs: List[Any]) -> Optional[Dict[str, str]]:
    try:
        mapping = extract_city_landmark_map(source_docs)
        if not mapping:
            return None
        fav_city_raw = _get_favourite_city()
        if not fav_city_raw:
            return None
        fav_city_key = _normalize_city_key(fav_city_raw)
        landmark = mapping.get(fav_city_key)
        if not landmark:
            return None
        lm = landmark.lower()
        if "gateway of india" in lm:
            endpoint = "getFirstCityFlightNumber"
        elif "taj mahal" in lm:
            endpoint = "getSecondCityFlightNumber"
        elif "eiffel tower" in lm:
            endpoint = "getThirdCityFlightNumber"
        elif "big ben" in lm:
            endpoint = "getFourthCityFlightNumber"
        else:
            endpoint = "getFifthCityFlightNumber"
        resp = _retry_request(f"https://register.hackrx.in/teams/public/flights/{endpoint}", timeout=REQUEST_TIMEOUT)
        try:
            j = resp.json()
            if isinstance(j, str):
                flight = j
            elif isinstance(j, dict):
                flight = j.get("flightNumber") or next(iter(j.values()), None)
            else:
                flight = resp.text.strip()
        except Exception:
            flight = resp.text.strip()
        if not flight:
            return None
        return {"favorite_city": fav_city_raw, "landmark": landmark, "flight_number": str(flight).strip()}
    except Exception:
        return None

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest, authorization: Optional[str] = Header(None)):
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization")
        token = authorization.split("Bearer ")[-1]
        if token != HACKRX_BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

    if not req.documents:
        raise HTTPException(status_code=400, detail="Missing 'documents'")

    tmp_path: Optional[str] = None
    try:
        _ensure_deps_for_pdf_and_requests()
        resp = _retry_request(req.documents, stream=True, timeout=PDF_STREAM_TIMEOUT)
        content_type = detect_content_type(resp, req.documents)
        if ".pdf" in req.documents.lower() or "pdf" in content_type:
            tmp_path = save_stream(resp, suffix=".pdf")
            vs, chunks = build_faiss_from_pdf(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Only PDF supported")

        flight_info = resolve_flight_number_from_docs(vs, chunks)
        if flight_info:
            return {"status": "success", **flight_info}
        return {"status": "error", "message": "Flight number not found"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                logger.warning("Failed to remove temp file %s", tmp_path)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
