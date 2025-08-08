"""
create_index.py

Utilities to:
- Download PDFs (streamed)
- Extract pages using PyMuPDF (fast for large PDFs)
- Do header/section-aware splitting
- Perform two-stage indexing: page-level index (coarse) and chunk-level index (fine)
- Batch embed text via OpenAIEmbeddings (LangChain wrapper)
- Build FAISS vector stores (ephemeral; not saved by default)

Primary exposed functions:
- build_page_index_from_pdf_sources(sources: List[str]) -> (page_index, pages_list)
- build_chunk_index_from_selected_pages(selected_page_texts: List[str]) -> chunk_index
"""

import os
import tempfile
import time
from typing import List, Tuple, Optional
import requests

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Read OPENAI key from env (ensure .env has it)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment. Put it into .env before running.")

# Embedding model you're allowed to use (tweak if you have different model)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Tunables for huge documents
PAGE_MIN_CHARS = 80           # drop pages under this length early
PAGE_LEVEL_CAP = 3000         # Max pages to embed at coarse page-level
PAGE_LEVEL_TOP_K = 20         # top pages per question from page-level index
MAX_SELECTED_PAGES = 600      # cap of pages to expand into chunks
MAX_CHUNKS = 3000             # cap final chunks to avoid explosion

# Chunking defaults (character based)
CHUNK_SIZE_SMALL = 800
CHUNK_SIZE_MEDIUM = 1000
CHUNK_SIZE_LARGE = 1200
CHUNK_OVERLAP_RATIO = 0.12

# Batch size for embedding calls (LangChain wrapper batches internally but good to keep)
EMBED_BATCH_SIZE = 64


# ---------------------------
# Low-level helpers
# ---------------------------
def download_pdf_to_temp(url: str, timeout: int = 60) -> str:
    """Stream-download URL into a temporary file and return the local path."""
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def load_pages_with_pymupdf(path: str):
    """
    Uses PyMuPDF loader (LangChain's PyMuPDFLoader) to extract page Documents.
    Returns a list of Document-like objects with `.page_content` and `.metadata`.
    """
    loader = PyMuPDFLoader(path)
    pages = loader.load()  # each page is a Document-like object
    return pages


def is_text_rich(text: Optional[str], min_chars: int = PAGE_MIN_CHARS) -> bool:
    return bool(text and len(text.strip()) >= min_chars)


def simple_header_split(text: str) -> List[str]:
    """
    Lightweight header-aware splitting:
    Splits when encountering lines that look like headers (ALL CAPS or "SECTION"/"CHAPTER" or numbered headings).
    Returns parts (strings) to be chunked further.
    """
    lines = text.splitlines()
    parts = []
    cur = []
    for ln in lines:
        s = ln.strip()
        if (s.isupper() and len(s) > 3) or s.startswith(("SECTION", "CHAPTER")) or (len(s) >= 2 and s[:2].isdigit()):
            if cur:
                parts.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        parts.append("\n".join(cur).strip())
    # filter empties
    return [p for p in parts if p and len(p.strip()) > 0]


# ---------------------------
# Two-stage index builders
# ---------------------------
def build_page_index_from_pdf_sources(sources: List[str]) -> Tuple[FAISS, List]:
    """
    Downloads & loads all pages from `sources` (URLs or file paths), filters tiny pages,
    and builds a page-level FAISS index (one vector per page). Returns (page_index, pages_selected).
    pages_selected is a list of Document-like objects (with page_content & metadata).
    """
    tmp_files = []
    all_pages = []
    try:
        # Download + load pages sequentially (keeps memory bounded)
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                tmp_path = download_pdf_to_temp(src)
                tmp_files.append(tmp_path)
                pages = load_pages_with_pymupdf(tmp_path)
                # attach source metadata
                for p in pages:
                    if not getattr(p, "metadata", None):
                        p.metadata = {}
                    p.metadata["source"] = src
            else:
                pages = load_pages_with_pymupdf(src)
                for p in pages:
                    if not getattr(p, "metadata", None):
                        p.metadata = {}
                    p.metadata["source"] = src
            # filter tiny pages
            pages = [p for p in pages if is_text_rich(getattr(p, "page_content", None))]
            all_pages.extend(pages)

        if not all_pages:
            raise ValueError("No usable text extracted from provided documents.")

        # score pages by length and cap
        pages_sorted = sorted(all_pages, key=lambda p: len(p.page_content or ""), reverse=True)
        pages_selected = pages_sorted[:min(len(pages_sorted), PAGE_LEVEL_CAP)]

        # instantiate embeddings and build FAISS index for pages (will embed page texts)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        page_index = FAISS.from_documents(pages_selected, embeddings)

        return page_index, pages_selected
    finally:
        # clean downloaded tmp files (we keep in-memory page objects)
        for f in tmp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass


def select_top_pages_for_questions(page_index: FAISS, questions: List[str], top_k: int = PAGE_LEVEL_TOP_K) -> List[str]:
    """
    For each question, query page_index and collect top page texts (deduped).
    Returns a deduped list of page_text strings (order preserved by first occurrence).
    """
    chosen_texts = []
    for q in questions:
        try:
            results = page_index.similarity_search(q, k=top_k)
        except Exception:
            # fallback if wrapper differs
            results = page_index.similarity_search(q, k=top_k)
        for doc in results:
            txt = getattr(doc, "page_content", None)
            if txt:
                chosen_texts.append(txt)
    # dedupe preserving order
    seen = set()
    dedup = []
    for t in chosen_texts:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    # cap to avoid explosion
    return dedup[:MAX_SELECTED_PAGES]


def build_chunk_index_from_selected_page_texts(selected_page_texts: List[str]) -> FAISS:
    """
    Given selected page texts (strings), apply header-aware pre-split then recursive char splitting
    to produce chunk documents. Then embed chunks (batched) and build FAISS chunk-level index.
    """
    # choose chunk_size adaptively
    total = len(selected_page_texts)
    if total <= 100:
        chunk_size = CHUNK_SIZE_SMALL
    elif total <= 300:
        chunk_size = CHUNK_SIZE_MEDIUM
    else:
        chunk_size = CHUNK_SIZE_LARGE
    chunk_overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)

    # header-aware pre-splitting
    pre_chunks = []
    for text in selected_page_texts:
        parts = simple_header_split(text)
        if parts:
            pre_chunks.extend(parts)
        else:
            pre_chunks.append(text)

    # wrap into lightweight docs and run recursive character splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " "]
    )

    # create pseudo-docs compatible with splitter: objects with page_content attribute
    class _Doc:
        def __init__(self, txt):
            self.page_content = txt

    wrapped = [_Doc(t) for t in pre_chunks]
    chunk_docs = splitter.split_documents(wrapped)

    # filter tiny chunks
    chunk_docs = [c for c in chunk_docs if len(getattr(c, "page_content", "").strip()) >= 200]

    # cap final chunks
    if len(chunk_docs) > MAX_CHUNKS:
        chunk_docs = chunk_docs[:MAX_CHUNKS]

    # build FAISS chunk-level index
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunk_index = FAISS.from_documents(chunk_docs, embeddings)

    return chunk_index
