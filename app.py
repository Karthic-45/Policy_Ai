#!/usr/bin/env python3
"""
HackRx Flight Number API — fixed, crash-resistant single-file implementation

Key improvements:
- Fixed logger initialization
- Robust error handling and clear JSON error responses (no uncaught crashes)
- Defensive parsing for HTTP responses and PDFs
- Safer handling of FAISS/vector store (returns both store and source docs)
- Cleaner temporary-file handling and guaranteed cleanup
- Configurable timeouts and retry settings via env
- Helpful logging at each major step to aid debugging

Note: this script still depends on third-party packages (requests, fitz/PyMuPDF,
langchain text splitter + embeddings and FAISS). If any optional dependency is
missing, the app will start but return descriptive errors for related operations.
"""

import os
import tempfile
import logging
import mimetypes
import time
import re
from typing import Optional, Iterable, Dict, Tuple, List, Any

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Try to import optional heavy deps and keep helpful flags
_HAS_REQUESTS = True
_HAS_FITZ = True
_HAS_LANGCHAIN = True
_HAS_FAISS = True
_HAS_EMBEDDINGS = True
try:
    import requests
except Exception:  # pragma: no cover - environment may lack libs
    _HAS_REQUESTS = False

try:
    import fitz  # PyMuPDF
except Exception:
    _HAS_FITZ = False

try:
    # langchain imports (may vary between versions). Wrap in try/except
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except Exception:
    _HAS_LANGCHAIN = False

# The FAISS & OpenAI embeddings import path can differ by environment.
# We'll import lazily where used and guard with clear error messages.

# -----------------------------
# Logging (fixed)
# -----------------------------
logger = logging.getLogger(_name_)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")

# Config (with sane defaults)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "8.0"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "2"))

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="HackRx Flight Number API", version="1.0.4-fixed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers and defensive wrappers
# -----------------------------

CLEAN_RE = re.compile(r"[^A-Za-z0-9\s]")

class DependencyError(Exception):
    pass


def _ensure_deps_for_pdf_and_requests():
    if not _HAS_REQUESTS:
        raise DependencyError("Missing dependency: requests. Install via 'pip install requests'.")
    if not _HAS_FITZ:
        raise DependencyError("Missing dependency: PyMuPDF (fitz). Install via 'pip install pymupdf'.")
    if not _HAS_LANGCHAIN:
        raise DependencyError("Missing dependency: langchain (text splitter). Install via 'pip install langchain'.")


def _retry_request(url: str, stream=False, timeout: float = REQUEST_TIMEOUT, retries: int = DOWNLOAD_RETRIES) -> requests.Response:
    if not _HAS_REQUESTS:
        raise DependencyError("requests not available")

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
            # backoff but keep it short
            time.sleep(0.5 * (2 ** attempt))
    # raise the last exception for caller to format response
    raise last_exc


def detect_content_type(resp: requests.Response, url: str) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"


class TemporaryFile:
    """Context manager wrapper for NamedTemporaryFile that guarantees cleanup."""
    def _init_(self, suffix: str = ""):
        self.suffix = suffix
        self.name = None

    def _enter_(self):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=self.suffix)
        self.name = tf.name
        tf.close()
        return self.name

    def _exit_(self, exc_type, exc, tb):
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
        raise DependencyError("PyMuPDF (fitz) not available")
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            text = doc.load_page(pno).get_text("text") or ""
            text = text.strip()
            if text:
                # Return simple dict if langchain Document not available
                if _HAS_LANGCHAIN:
                    yield Document(page_content=text, metadata={"page": pno + 1})
                else:
                    yield {"page_content": text, "metadata": {"page": pno + 1}}
    finally:
        doc.close()


def build_faiss_from_pdf(pdf_path: str) -> Tuple[Any, List[Any]]:
    """Builds FAISS vector store and returns (vector_store, source_docs_list).
    If FAISS or embeddings are not available, raises DependencyError with instructions.
    """
    # Ensure deps
    if not _HAS_LANGCHAIN or not _HAS_FITZ or not _HAS_REQUESTS:
        raise DependencyError("Missing required libraries to build vector store (langchain/fitz/requests).")

    # Lazy imports for FAISS & embeddings because their import may fail in some envs
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        raise DependencyError("Missing FAISS vector store. Install pip install langchain-community or an equivalent that provides FAISS.")

    try:
        # Try multiple import paths for OpenAI embeddings wrapper
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception:
            # fallback if package name differs in your environment
            from langchain.embeddings import OpenAIEmbeddings
    except Exception:
        raise DependencyError("Missing OpenAI embeddings. Install pip install openai langchain and set OPENAI_API_KEY environment variable.")

    if not OPENAI_API_KEY:
        raise DependencyError("OPENAI_API_KEY is not set. Set it in the environment or .env file.")

    # create embeddings client
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # chunk splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=CHUNK_OVERLAP
    )

    docs = list(iter_pdf_pages(pdf_path))
    if not docs:
        raise ValueError("No text could be extracted from PDF.")

    # use langchain splitter if available
    if hasattr(splitter, "split_documents"):
        chunks = splitter.split_documents(docs)
    else:
        # fallback — naive chunker
        chunks = []
        for d in docs:
            text = d.page_content if hasattr(d, "page_content") else d["page_content"]
            for i in range(0, len(text), 1000):
                chunk_text = text[i:i + 1200]
                chunks.append(Document(page_content=chunk_text, metadata=getattr(d, "metadata", d.get("metadata", {}))))

    chunks = [c for c in chunks if len((c.page_content if hasattr(c, "page_content") else c["page_content"]).strip()) >= MIN_CHUNK_LEN]
    if not chunks:
        raise ValueError("No valid chunks after splitting PDF.")

    vs = FAISS.from_documents(chunks, embeddings)
    # return both vector store and the original chunks so we don't rely on internal attributes
    return vs, chunks

# -----------------------------
# Mapping extraction (robust)
# -----------------------------

def _normalize_city_key(s: str) -> str:
    s = (s or "").strip()
    s = CLEAN_RE.sub("", s)  # remove emojis/punctuation
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def extract_city_landmark_map(docs) -> Dict[str, str]:
    """Build mapping: normalized_city -> landmark (original-cased).
    Works with either langchain Document objects or simple dicts with page_content.
    """
    mapping: Dict[str, str] = {}
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("page_content", "")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # remove leading/trailing punctuation but keep words
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
    logger.info("Extracted %d city->landmark entries from document", len(mapping))
    return mapping

# -----------------------------
# Favourite city & flight resolution
# -----------------------------

def _get_favourite_city() -> Optional[str]:
    if not _HAS_REQUESTS:
        logger.warning("requests not available; cannot fetch favourite city")
        return None
    try:
        resp = _retry_request("https://register.hackrx.in/submissions/myFavouriteCity", timeout=REQUEST_TIMEOUT)
        # try JSON then text
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
    except Exception as e:
        logger.warning("Failed to fetch favourite city: %s", e)
        return None


def resolve_flight_number_from_docs(vector_store: Any, source_docs: List[Any]) -> Optional[Dict[str, str]]:
    """Given a vector_store and the original chunks/docs, attempt to resolve flight number.
    This function is deliberately defensive and returns None if any step can't be completed.
    """
    try:
        mapping = extract_city_landmark_map(source_docs)
        if not mapping:
            logger.warning("No mapping found in extracted documents")
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
                if isinstance(j, str):
                    flight = j
                elif isinstance(j, dict):
                    flight = j.get("flightNumber") or next(iter(j.values()), None)
                else:
                    flight = resp.text.strip()
            except Exception:
                flight = resp.text.strip()
            if not flight:
                logger.warning("Flight endpoint %s returned empty body.", endpoint)
                return None
            return {"favorite_city": fav_city_raw, "landmark": landmark, "flight_number": str(flight).strip()}
        except Exception as e:
            logger.exception("Error fetching flight number from endpoint: %s", e)
            return None
    except Exception as e:
        logger.exception("Unhandled error in resolve_flight_number_from_docs: %s", e)
        return None

# -----------------------------
# Endpoint (defensive)
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(
    url: str = Query(..., description="PDF file URL"),
    authorization: Optional[str] = Header(None),
):
    # Authorization check (optional if env not set)
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization")
        token = authorization.split("Bearer ")[-1]
        if token != HACKRX_BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' query parameter")

    tmp_path: Optional[str] = None
    try:
        # ensure required libs for download & PDF
        try:
            _ensure_deps_for_pdf_and_requests()
        except DependencyError as de:
            logger.error(str(de))
            raise HTTPException(status_code=500, detail=str(de))

        # download
        try:
            resp = _retry_request(url, stream=True, timeout=PDF_STREAM_TIMEOUT)
        except Exception as e:
            logger.exception("Failed to download URL: %s", e)
            raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")

        content_type = detect_content_type(resp, url)
        logger.info("Content-Type detected: %s", content_type)

        if ".pdf" in url.lower() or "pdf" in content_type:
            # save to a temp file and make sure it's cleaned
            tmp_path = save_stream(resp, suffix=".pdf")
            logger.info("Saved remote PDF to %s", tmp_path)

            try:
                vs, chunks = build_faiss_from_pdf(tmp_path)
            except DependencyError as de:
                # propagate clear error for missing deps
                logger.error(str(de))
                raise HTTPException(status_code=500, detail=str(de))
            except ValueError as ve:
                logger.warning("PDF processing issue: %s", ve)
                return {"status": "error", "step": "pdf_processing", "message": str(ve)}
            except Exception as e:
                logger.exception("Unexpected error building vector store: %s", e)
                raise HTTPException(status_code=500, detail="Unexpected error while processing PDF")
        else:
            raise HTTPException(status_code=400, detail="Only PDF supported for this task.")

        # attempt to resolve flight number
        flight_info = resolve_flight_number_from_docs(vs, chunks)
        if flight_info:
            return {"status": "success", **flight_info}

        # If control reaches here, we couldn't resolve the flight
        return {"status": "error", "message": "Flight number not found in document"}

    except HTTPException:
        # re-raise HTTPExceptions so FastAPI can turn them into responses
        raise
    except Exception as e:
        logger.exception("Internal server error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # cleanup temp file if any
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug("Deleted temp file %s", tmp_path)
            except Exception:
                logger.warning("Failed to remove temp file %s", tmp_path)


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
