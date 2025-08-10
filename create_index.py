import os
import sys
import time
import mimetypes
import zipfile
import tempfile
import logging
import argparse
from typing import List

try:
    import magic
except ImportError:
    magic = None

import pytesseract
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Environment Setup
# ---------------------------
def load_env(env_path: str = ".env") -> None:
    """Load environment variables from a .env file."""
    load_dotenv(dotenv_path=env_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in .env file.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = openai_api_key

# ---------------------------
# File Type Detection
# ---------------------------
def detect_file_type(file_path: str) -> str:
    """Detect file extension by magic numbers and mimetypes."""
    if magic:
        try:
            mime_type = magic.from_file(file_path, mime=True)
            ext = mimetypes.guess_extension(mime_type) or ""
        except Exception as e:
            logger.warning(f"magic failed: {e}")
            ext = ""
    else:
        ext = mimetypes.guess_extension(mimetypes.guess_type(file_path)[0] or '') or ""

    with open(file_path, "rb") as f:
        sig = f.read(8)

    if sig.startswith(b"%PDF-"):
        return ".pdf"
    if sig.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(file_path) as z:
            names = z.namelist()
            if any(n.startswith("word/") for n in names):
                return ".docx"
            if any(n.startswith("xl/") for n in names):
                return ".xlsx"
            if any(n.startswith("ppt/") for n in names):
                return ".pptx"
        return ".zip"
    if sig.startswith(b"\xFF\xD8\xFF"):
        return ".jpg"
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return ext or ".txt"

# ---------------------------
# Universal Loader
# ---------------------------
def load_document(file_path: str) -> List[Document]:
    """Load a document and extract its content as Document(s)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ["", ".bin"]:
        ext = detect_file_type(file_path)

    try:
        if ext == ".pdf":
            return PyMuPDFLoader(file_path).load()
        elif ext in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(file_path).load()
        elif ext == ".pptx":
            return UnstructuredPowerPointLoader(file_path).load()
        elif ext in [".html", ".htm"]:
            return UnstructuredHTMLLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".csv":
            return CSVLoader(file_path).load()
        elif ext == ".xlsx":
            return UnstructuredExcelLoader(file_path).load()
        elif ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return [Document(page_content=text, metadata={"source": file_path})]
        elif ext == ".zip":
            docs = []
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        docs.extend(load_document(os.path.join(root, f)))
            return docs
        else:
            # Fallback: try reading as text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                return [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                logger.warning(f"Cannot read {file_path} as text: {e}")
                return []
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return []

# ---------------------------
# Load All Documents (with progress bar)
# ---------------------------
def load_all_documents(folder_path: str) -> List[Document]:
    """Load all documents from a folder recursively."""
    all_docs = []
    if not os.path.exists(folder_path):
        logger.error(f"Folder '{folder_path}' not found.")
        return []

    files_to_load = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            files_to_load.append(os.path.join(root, filename))

    for file_path in tqdm(files_to_load, desc="Loading documents"):
        docs = load_document(file_path)
        if docs:
            all_docs.extend(docs)
            logger.info(f"Loaded: {os.path.basename(file_path)}")
        else:
            logger.warning(f"Skipped: {os.path.basename(file_path)}")
    return all_docs

# ---------------------------
# Chunking
# ---------------------------
def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Chunk the documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ---------------------------
# Create FAISS Index
# ---------------------------
def create_and_save_index(chunks: List[Document], index_path: str) -> None:
    """Create FAISS index and save locally."""
    logger.info("Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    logger.info("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    logger.info(f"Index saved at: {index_path}")

# ---------------------------
# Argument Parsing
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Index documents into FAISS vector store.")
    parser.add_argument("--docs", type=str, default="./documents/", help="Path to documents folder")
    parser.add_argument("--index", type=str, default="faiss_universal_index", help="Path to save FAISS index")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each text chunk")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between chunks")
    return parser.parse_args()

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    load_env(args.env)
    start_time = time.time()

    logger.info("Loading documents...")
    docs = load_all_documents(args.docs)

    if docs:
        logger.info("Splitting into chunks...")
        chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info("Saving FAISS index...")
        create_and_save_index(chunks, args.index)
    else:
        logger.warning("No documents found.")

    logger.info(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
