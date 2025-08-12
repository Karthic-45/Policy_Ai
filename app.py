import os
import time
import tempfile
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PIL import Image
import logging

# ---------- CONFIG ----------
PDF_STREAM_TIMEOUT = 30
MIN_CHUNK_LEN = 5
CHUNK_OVERLAP = 50
MAX_CHUNKS = 100
embeddings = None  # Replace with your embeddings instance

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ---------- REQUEST MODEL ----------
class HackrxRequest(BaseModel):
    documents: str

# ---------- RETRY REQUEST ----------
def _retry_request(url: str, stream: bool = False, timeout: int = 30, retries: int = 3, backoff: int = 2) -> requests.Response:
    """
    Retry HTTP requests with exponential backoff.
    Images get more retries and longer timeout automatically.
    """
    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
    if any(url.lower().endswith(ext) for ext in image_exts):
        retries = 5
        timeout = max(timeout, 60)

    last_exc = None
    for attempt in range(retries):
        try:
            logger.info("Downloading (%d/%d): %s", attempt + 1, retries, url)
            resp = requests.get(url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 409:
                logger.error("File is private or requires authentication: %s", url)
                raise HTTPException(status_code=403, detail="The file is private or requires authentication.")
            last_exc = e
        except Exception as e:
            last_exc = e
            logger.warning("Download attempt %d failed: %s", attempt + 1, e)
        time.sleep(backoff * (2 ** attempt))
    logger.error("All download attempts failed for %s", url)
    raise last_exc

# ---------- /hackrx/run ----------
@app.post("/hackrx/run")
async def hackrx_run(data: HackrxRequest):
    try:
        logger.info("Received /hackrx/run for document: %s", data.documents)

        # Download file with retry
        resp = _retry_request(data.documents, stream=True, timeout=PDF_STREAM_TIMEOUT)
        tmp_fd, tmp_path = tempfile.mkstemp()
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp_file.write(chunk)

        ext = Path(tmp_path).suffix.lower()
        logger.info("Downloaded file to %s (ext=%s)", tmp_path, ext)

        # -------- File type handling --------
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        elif ext in [".zip", ".rar"]:
            loader = UnstructuredFileLoader(tmp_path)
            docs = loader.load()

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]:
            try:
                logger.info("Detected image file: %s", tmp_path)
                try:
                    docs = UnstructuredImageLoader(tmp_path).load()
                    logger.info("Extracted text from image using UnstructuredImageLoader.")
                except Exception as e:
                    logger.warning("OCR extraction failed, falling back to metadata: %s", e)
                    with Image.open(tmp_path) as img:
                        meta_info = f"Image metadata: format={img.format}, size={img.size}, mode={img.mode}"
                        docs = [Document(page_content=meta_info)]
            except Exception as e:
                logger.exception("Failed to process image file: %s", e)
                raise HTTPException(status_code=400, detail="Failed to read image file.")

            # Filter & chunk for images
            docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= MIN_CHUNK_LEN]
            if not docs:
                docs = [Document(page_content="[EMPTY IMAGE CONTENT]")]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = chunks[:MAX_CHUNKS]
            FAISS.from_documents(chunks, embeddings)
            return {"status": "success", "chunks": len(chunks)}

        else:
            loader = UnstructuredFileLoader(tmp_path)
            docs = loader.load()

        # -------- Filter & chunk for all other file types --------
        docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= MIN_CHUNK_LEN]
        if not docs:
            return {"status": "error", "message": "No readable content found."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        chunks = chunks[:MAX_CHUNKS]
        FAISS.from_documents(chunks, embeddings)

        return {"status": "success", "chunks": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in /hackrx/run: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
