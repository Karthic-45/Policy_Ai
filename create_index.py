#!/usr/bin/env python3
"""
create_index.py

Usage examples:
  python create_index.py --input-folder ./policy_docs --index-path ./faiss_index
  python create_index.py --input-file https://.../policy.pdf --index-path ./faiss_index

Outputs:
  - FAISS index files in --index-path
  - metadata_map.json (maps chunk ids -> source/metadata)
"""

import os
import re
import json
import tempfile
import zipfile
import mimetypes
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import argparse
import logging

# LangChain & OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Document loaders (try to import; if not installed, user must pip install proper packages)
try:
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader, UnstructuredImageLoader
except Exception as e:
    raise ImportError("Please install langchain-community and optional loader deps (unstructured, pymupdf). Error: %s" % e)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_index")

# ---------- Utilities ----------
def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

def download_to_temp(url: str) -> str:
    import requests
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    suffix = mimetypes.guess_extension(r.headers.get("content-type", "")) or Path(url).suffix or ".bin"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(r.content)
    tf.flush()
    tf.close()
    return tf.name

def list_files_recursive(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files

# ---------- Loader ----------
def load_documents_from_path(path: str) -> List[Document]:
    """
    Load a single file path (or downloaded URL) into langchain Documents.
    Sets metadata['source'] to original path.
    """
    docs: List[Document] = []
    orig_path = path
    if is_url(path):
        path = download_to_temp(path)

    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            loader = PyMuPDFLoader(path)
            docs = loader.load()
        elif ext in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(path)
            docs = loader.load()
        elif ext in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(path)
            docs = loader.load()
        elif ext in [".eml", ".msg"]:
            loader = UnstructuredEmailLoader(path)
            docs = loader.load()
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            loader = UnstructuredImageLoader(path)
            docs = loader.load()
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            docs = [Document(page_content=df.to_string(), metadata={"source": orig_path})]
        elif ext == ".xlsx":
            import pandas as pd
            df = pd.read_excel(path)
            docs = [Document(page_content=df.to_string(), metadata={"source": orig_path})]
        elif ext == ".zip":
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(path, "r") as z:
                    z.extractall(extract_dir)
                for root, _, files in os.walk(extract_dir):
                    for f in files:
                        full = os.path.join(root, f)
                        docs.extend(load_documents_from_path(full))
        else:
            # fallback: attempt word then pdf then load raw text
            try:
                loader = UnstructuredWordDocumentLoader(path)
                docs = loader.load()
            except Exception:
                try:
                    loader = PyMuPDFLoader(path)
                    docs = loader.load()
                except Exception:
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            text = fh.read()
                            docs = [Document(page_content=text, metadata={"source": orig_path})]
                    except Exception as e:
                        logger.warning(f"Could not load {path}: {e}")
                        docs = []
    except Exception as e:
        logger.warning(f"Loader failure for {path}: {e}")
        docs = []

    # Normalize metadata
    for d in docs:
        md = dict(d.metadata or {})
        md["source"] = md.get("source", orig_path)
        d.metadata = md
    return docs

# ---------- Heading-aware structure split ----------
HEADER_RE = re.compile(r"^\s*(?:section|clause|article|chapter|heading|part|section\s*\d+|[A-Z][A-Za-z0-9\s]{1,60})\s*[:\-\n]", re.I)
NUMBERED_LINE = re.compile(r"^\s*(\d+[\.\)]|\(\d+\)|[ivx]+\.)\s+", re.I)

def structure_aware_split(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Breaks documents into logical blocks by headings and blank lines, then uses a RecursiveCharacterTextSplitter.
    Returns list[Document] with metadata including block_id and chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    blocks: List[Document] = []
    for doc_idx, doc in enumerate(documents):
        text = doc.page_content or ""
        source = doc.metadata.get("source", f"doc_{doc_idx}")
        # split by two or more newlines (paragraph/section separation)
        raw_paras = re.split(r"\n{2,}", text)
        block_id = 0
        for para in raw_paras:
            para = para.strip()
            if not para:
                continue
            # If para contains heading markers near start, treat as a block boundary
            lines = para.splitlines()
            current_lines = []
            for line in lines:
                # treat heading or numbered bullet as block boundary
                if HEADER_RE.match(line.strip()) or NUMBERED_LINE.match(line.strip()):
                    if current_lines:
                        block_text = "\n".join(current_lines).strip()
                        if block_text:
                            blocks.append(Document(page_content=block_text, metadata={**doc.metadata, "source": source, "block_id": str(block_id)}))
                            block_id += 1
                            current_lines = []
                    current_lines.append(line)
                else:
                    current_lines.append(line)
            if current_lines:
                block_text = "\n".join(current_lines).strip()
                if block_text:
                    blocks.append(Document(page_content=block_text, metadata={**doc.metadata, "source": source, "block_id": str(block_id)}))
                    block_id += 1

    # Now split blocks into chunks
    final_chunks: List[Document] = []
    for b_idx, block in enumerate(blocks):
        pieces = splitter.split_documents([block])
        for i, piece in enumerate(pieces):
            md = dict(piece.metadata or {})
            md["block_id"] = md.get("block_id", str(b_idx))
            md["chunk_id"] = f"{md['block_id']}_{i}"
            final_chunks.append(Document(page_content=piece.page_content, metadata=md))
    return final_chunks

# ---------- Adaptive chunking ----------
def determine_chunk_params(documents: List[Document], target_max_chunks=300, model_context_tokens=8192) -> Tuple[int, int]:
    total_chars = sum(len(d.page_content or "") for d in documents)
    page_count = max(1, len(documents))
    avg_chars_per_page = total_chars / page_count

    # base
    if page_count <= 3:
        base = 700
    elif page_count <= 10:
        base = 600
    else:
        base = 450

    if avg_chars_per_page > 3000:
        base = min(base + 300, 1500)
    elif avg_chars_per_page < 800:
        base = max(base - 200, 300)

    est_chunks = max(1, total_chars // base)
    if est_chunks > target_max_chunks:
        base = int(total_chars / target_max_chunks) + 50

    overlap = 150 if base <= 500 else 80 if base <= 900 else 50
    return base, overlap

# ---------- Build & Save FAISS ----------
def build_faiss_index(documents: List[Document], index_path: str, embedding_model: str = "text-embedding-ada-002"):
    if not documents:
        raise ValueError("No documents to index.")
    chunk_size, chunk_overlap = determine_chunk_params(documents)
    logger.info(f"chunk_size={chunk_size}, overlap={chunk_overlap}, documents={len(documents)}")

    logger.info("Structure-aware splitting...")
    chunks = structure_aware_split(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # filter tiny chunks
    chunks = [c for c in chunks if len((c.page_content or "").strip()) > 50]
    logger.info(f"Total chunks after split/filter: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model=embedding_model)
    logger.info("Creating FAISS vectorstore (this will batch embeddings automatically)...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    logger.info(f"FAISS saved to {index_path}")

    # Save chunk metadata mapping
    meta_map = []
    for c in chunks:
        md = c.metadata or {}
        meta_map.append({
            "chunk_id": md.get("chunk_id"),
            "block_id": md.get("block_id"),
            "source": md.get("source"),
            "excerpt": (c.page_content or "")[:800]
        })
    with open(os.path.join(index_path, "metadata_map.json"), "w", encoding="utf-8") as fh:
        json.dump(meta_map, fh, indent=2)
    logger.info("Saved metadata_map.json")
    return vector_store

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default=None, help="Folder containing files to index")
    parser.add_argument("--input-file", type=str, default=None, help="Single file path or URL to index")
    parser.add_argument("--index-path", type=str, default="faiss_index", help="Folder to save FAISS index")
    parser.add_argument("--embedding-model", type=str, default=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    args = parser.parse_args()

    sources = []
    if args.input_folder:
        sources = list_files_recursive(args.input_folder)
    elif args.input_file:
        sources = [args.input_file]
    else:
        raise SystemExit("Provide --input-folder or --input-file")

    all_docs: List[Document] = []
    for s in sources:
        try:
            loaded = load_documents_from_path(s)
            all_docs.extend(loaded)
            logger.info(f"Loaded {len(loaded)} docs from {s}")
        except Exception as e:
            logger.warning(f"Error loading {s}: {e}")

    if not all_docs:
        raise SystemExit("No documents loaded; aborting.")

    build_faiss_index(all_docs, args.index_path, embedding_model=args.embedding_model)
    logger.info("Indexing complete.")

if __name__ == "__main__":
    main()
