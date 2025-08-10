#!/usr/bin/env python3
"""
HackRx RAG backend with Flight Number Resolver (URL-based) — improved/resilient version
"""

import os
import tempfile
import logging
import mimetypes
import time
import re
from typing import Optional, Iterable, Dict

import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")

# Config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "2"))

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="HackRx Flight Number API", version="1.0.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"]
)

# -----------------------------
# Models
# -----------------------------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# -----------------------------
# Helpers
# -----------------------------
def _retry_request(url: str, stream=False, timeout=REQUEST_TIMEOUT, retries=DOWNLOAD_RETRIES) -> requests.Response:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            logger.debug("HTTP request attempt %d -> %s", attempt + 1, url)
            resp = requests.get(url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            logger.warning("Request attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(0.5 * (2 ** attempt))
    raise last_exc

def detect_content_type(resp: requests.Response, url: str) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"

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

def iter_pdf_pages(pdf_path: str) -> Iterable[Document]:
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            text = doc.load_page(pno).get_text("text") or ""
            text = text.strip()
            if text:
                yield Document(page_content=text, metadata={"page": pno + 1})
    finally:
        doc.close()

def build_faiss_from_pdf(pdf_path: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=CHUNK_OVERLAP
    )
    docs = list(iter_pdf_pages(pdf_path))
    if not docs:
        raise ValueError("No text could be extracted from PDF.")
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    if not chunks:
        raise ValueError("No valid chunks after splitting PDF.")
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# Mapping extraction (robust)
# -----------------------------
CLEAN_RE = re.compile(r"[^A-Za-z0-9\s]")

def _normalize_city_key(s: str) -> str:
    s = s.strip()
    s = CLEAN_RE.sub("", s)  # remove emojis/punctuation
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def extract_city_landmark_map(docs) -> Dict[str, str]:
    """
    Build mapping: normalized_city -> landmark (original-cased).
    Tries to parse lines where last token is city and preceding part is landmark.
    """
    mapping = {}
    for doc in docs:
        for raw_line in doc.page_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # remove leading emojis/symbols
            cleaned = CLEAN_RE.sub("", line).strip()
            parts = cleaned.split()
            if len(parts) < 2:
                continue
            city_candidate = parts[-1]
            landmark_candidate = " ".join(parts[:-1]).strip()
            city_key = _normalize_city_key(city_candidate)
            if not city_key:
                continue
            # store landmark in original-ish cleaned form
            mapping[city_key] = landmark_candidate
    logger.info("Extracted %d city->landmark entries from document", len(mapping))
    return mapping

# -----------------------------
# Flight resolution
# -----------------------------
def _get_favourite_city() -> Optional[str]:
    try:
        resp = _retry_request("https://register.hackrx.in/submissions/myFavouriteCity", timeout=REQUEST_TIMEOUT)
        # try JSON then text
        try:
            j = resp.json()
            if isinstance(j, (str,)):
                city = j
            elif isinstance(j, dict):
                # try common keys
                city = j.get("city") or j.get("favouriteCity") or next(iter(j.values()), None)
            else:
                city = None
        except Exception:
            city = resp.text.strip()
        if not city:
            return None
        return CLEAN_RE.sub("", str(city)).strip()
    except Exception as e:
        logger.warning("Failed to fetch favourite city: %s", e)
        return None

def resolve_flight_number_from_docs(vector_store: FAISS):
    docs_list = list(vector_store.docstore._dict.values())
    mapping = extract_city_landmark_map(docs_list)
    if not mapping:
        return None

    fav_city_raw = _get_favourite_city()
    if not fav_city_raw:
        logger.warning("Favourite city API returned nothing.")
        return None
    fav_city_key = _normalize_city_key(fav_city_raw)

    landmark = mapping.get(fav_city_key)
    if not landmark:
        logger.warning("City '%s' (%s) not found in mapping.", fav_city_raw, fav_city_key)
        return None

    # map to endpoints (case-insensitive match of landmark)
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

    try:
        resp = _retry_request(f"https://register.hackrx.in/teams/public/flights/{endpoint}", timeout=REQUEST_TIMEOUT)
        try:
            j = resp.json()
            if isinstance(j, (str,)):
                flight = j
            elif isinstance(j, dict):
                flight = j.get("flightNumber") or next(iter(j.values()), None)
            else:
                flight = resp.text.strip()
        except Exception:
            flight = resp.text.strip()
        if not flight:
            logger.warning("Flight endpoint returned empty body.")
            return None
        return {"favorite_city": fav_city_raw, "landmark": landmark, "flight_number": str(flight).strip()}
    except Exception as e:
        logger.exception("Error fetching flight number from endpoint: %s", e)
        return None

# -----------------------------
# Endpoint
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(
    url: str = Query(..., description="PDF file URL"),
    authorization: Optional[str] = Header(None)
):
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization")
        token = authorization.split("Bearer ")[1]
        if token != HACKRX_BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

    tmp_path = None
    try:
        resp = _retry_request(url, stream=True, timeout=PDF_STREAM_TIMEOUT)
        content_type = detect_content_type(resp, url)

        if ".pdf" in url.lower() or "pdf" in content_type:
            tmp_path = save_stream(resp, suffix=".pdf")
            vector_store = build_faiss_from_pdf(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Only PDF supported for this task.")

        flight_info = resolve_flight_number_from_docs(vector_store)
        if flight_info:
            return {"status": "success", **flight_info}

        return {"status": "error", "message": "Flight number not found in document"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
