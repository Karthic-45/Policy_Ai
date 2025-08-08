

import os
import tempfile
import time
import requests
import asyncio
from typing import List, Optional, Tuple, Dict
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain-ish imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="HackRx MultiPDF Q&A (Full)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# TUNABLE CONSTANTS
# -------------------
# models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # change to your preferred embedding model
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")  # you said GPT-4

# pipeline limits (tune for infra/timeouts)
PAGE_MIN_CHARS = 80           # drop pages under this length
PAGE_LEVEL_TOP_K = 20         # top pages per question at page-level
PAGE_LEVEL_CAP = 3000         # max pages to embed at page-level (global cap)
MAX_SELECTED_PAGES = 600      # after page-level picks, cap selected pages to this
MAX_CHUNKS = 3000             # cap chunk docs to this
CHUNK_SIZE_SMALL = 800
CHUNK_SIZE_MEDIUM = 1000
CHUNK_SIZE_LARGE = 1200
CHUNK_OVERLAP_PCT = 0.12
EMBED_BATCH_SIZE = 64

# prompt
PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Use ONLY the provided document excerpts.
•⁠  ⁠QUOTE exact policy text in double quotes when you use it.
•⁠  ⁠If multiple excerpts are needed, synthesize them concisely and include quoted passages.
•⁠  ⁠Do NOT guess. If not present: "The policy document does not specify this clearly."
•⁠  ⁠Keep answers short and direct unless the question asks for explanation.

Context:
{context}

Question: {input}
Answer:
"""

# instantiate models once
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0)
PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)
qa_chain = create_stuff_documents_chain(llm, PROMPT)


# -------------------
# Request models
# -------------------
class HackRxRequest(BaseModel):
    documents: List[str]   # list of URLs. If using file uploads, use /upload_and_run route instead
    questions: List[str]


# -------------------
# Helper utilities
# -------------------
def is_text_rich(text: str, min_chars: int = PAGE_MIN_CHARS) -> bool:
    return bool(text and len(text.strip()) >= min_chars)


def simple_header_split(text: str) -> List[str]:
    """
    Naive header-aware split: split when encountering lines that look like headers.
    This keeps sections together prior to recursive splitting.
    """
    lines = text.splitlines()
    pieces = []
    current = []
    for ln in lines:
        stripped = ln.strip()
        if (stripped.isupper() and len(stripped) > 3) or stripped.startswith(("SECTION", "CHAPTER")) or (len(stripped) > 0 and stripped[:3].isdigit()):
            if current:
                pieces.append("\n".join(current).strip())
            current = [stripped]
        else:
            current.append(ln)
    if current:
        pieces.append("\n".join(current).strip())
    return [p for p in pieces if p]


def chunk_texts(texts: List[str], chunk_size: int, chunk_overlap: int):
    """
    Use RecursiveCharacterTextSplitter to produce chunk Documents (LangChain-style)
    Expects texts = list of strings (page texts or section texts).
    Returns list of pseudo-Document objects with .page_content attribute for FAISS.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    # create small wrapper class so splitter.split_documents works (expects Document-like)
    class DocWrapper:
        def _init_(self, text, meta=None):
            self.page_content = text
            self.metadata = meta or {}

    wrapped = [DocWrapper(t) for t in texts]
    chunks = splitter.split_documents(wrapped)
    return chunks


def adaptive_limits_by_selected_pages(total_selected_pages: int) -> Tuple[int, int]:
    """
    Decide chunk_size and chunk_overlap based on selected page count.
    """
    if total_selected_pages <= 100:
        return CHUNK_SIZE_SMALL, int(CHUNK_SIZE_SMALL * CHUNK_OVERLAP_PCT)
    elif total_selected_pages <= 300:
        return CHUNK_SIZE_MEDIUM, int(CHUNK_SIZE_MEDIUM * CHUNK_OVERLAP_PCT)
    else:
        return CHUNK_SIZE_LARGE, int(CHUNK_SIZE_LARGE * CHUNK_OVERLAP_PCT)


# -------------------
# Two-stage pipeline
# -------------------
def build_page_level_index(all_pages: List) -> Tuple[FAISS, List]:
    """
    Build a FAISS index of pages (one doc per page).
    Filters tiny pages and caps the number of pages embedded (PAGE_LEVEL_CAP).
    Returns (page_index, pages_selected)
    """
    # Filter tiny pages
    pages = [p for p in all_pages if is_text_rich(getattr(p, "page_content", ""))]
    if not pages:
        raise ValueError("No usable text pages found.")

    # Sort by length (proxy for information density)
    pages_sorted = sorted(pages, key=lambda p: len(p.page_content or ""), reverse=True)
    cap = min(len(pages_sorted), PAGE_LEVEL_CAP)
    pages_selected = pages_sorted[:cap]

    # Build FAISS (this will call embed_documents internally via LangChain wrapper)
    page_index = FAISS.from_documents(pages_selected, embeddings)
    return page_index, pages_selected


def select_top_pages(page_index: FAISS, questions: List[str], pages_selected: List, top_k_per_q: int = PAGE_LEVEL_TOP_K) -> List[str]:
    """
    For each question, perform a page-level similarity search and return deduplicated page texts.
    Returns list of page text strings (deduped).
    """
    chosen_texts = []
    for q in questions:
        try:
            results = page_index.similarity_search(q, k=top_k_per_q)
        except Exception:
            # fallback to default search if wrappers differ
            results = page_index.similarity_search(q, k=top_k_per_q)
        for doc in results:
            chosen_texts.append(doc.page_content)
    # dedupe while preserving order
    seen = set()
    deduped = []
    for t in chosen_texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    # cap
    return deduped[:MAX_SELECTED_PAGES]


async def answer_question_from_chunk_index(qa_chain, chunk_index: FAISS, question: str) -> str:
    """
    Query chunk-level index and call LLM chain to answer.
    Uses dynamic k for retrieval.
    """
    qlen = len(question.strip())
    k = 6 if qlen < 40 else (10 if qlen < 120 else 14)

    # get top chunks
    try:
        top_docs = chunk_index.similarity_search(question, k=k)
    except Exception:
        top_docs = chunk_index.similarity_search(question, k=k)

    # call chain (try async API)
    try:
        raw = await qa_chain.ainvoke({"context": top_docs, "input": question})
    except Exception:
        raw = qa_chain.invoke({"context": top_docs, "input": question})
    ans = raw.strip()
    if not ans or "i don't know" in ans.lower() or "does not specify" in ans.lower():
        return "The policy document does not specify this clearly."
    return ans


# -------------------
# Utilities: download & load pages
# -------------------
def download_to_tempfile(url: str, timeout: int = 60) -> str:
    """
    Stream-download URL to a temporary file; return file path.
    """
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return tmp.name


def load_pdf_pages_from_path(path: str):
    """
    Use PyMuPDFLoader to load pages. Returns list of Document-like objects (with page_content).
    """
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    return docs


# -------------------
# Main endpoint
# -------------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    """
    data.documents: list of PDF URLs (strings)
    data.questions: list of question strings
    """
    # auth optional
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if expected_token:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing/invalid Authorization header.")
        token = authorization.split("Bearer ")[1]
        if token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid token.")

    start_total = time.time()
    tmp_files = []
    all_pages = []

    try:
        # 1) Download & load pages from each PDF sequentially (keeps memory bounded)
        for url in data.documents:
            try:
                tmp_path = download_to_tempfile(url)
                tmp_files.append(tmp_path)
                pages = load_pdf_pages_from_path(tmp_path)
            except Exception as e:
                # cleanup previously created temp files before returning
                raise HTTPException(status_code=400, detail=f"Failed to download/load {url}: {e}")

            # keep only text-rich pages
            pages = [p for p in pages if is_text_rich(p.page_content)]
            # attach source metadata for provenance
            for p in pages:
                if not hasattr(p, "metadata") or p.metadata is None:
                    p.metadata = {}
                p.metadata["source_url"] = url
            all_pages.extend(pages)

        if not all_pages:
            raise HTTPException(status_code=400, detail="No usable text found in provided PDFs.")

        # 2) Build page-level index (coarse)
        page_index_build_start = time.time()
        page_index, pages_selected = build_page_level_index(all_pages)
        page_index_build_time = time.time() - page_index_build_start

        # 3) For each question, select top pages (dedup)
        selected_page_texts = select_top_pages(page_index, data.questions, pages_selected, top_k_per_q=PAGE_LEVEL_TOP_K)

        if not selected_page_texts:
            # nothing selected -> return generic "not specified" for each question
            answers = ["The policy document does not specify this clearly." for _ in data.questions]
            return {
                "status": "success",
                "answers": answers,
                "meta": {
                    "num_input_pdfs": len(data.documents),
                    "total_pages_found": len(all_pages),
                    "pages_used_for_page_index": len(pages_selected),
                    "selected_pages_for_chunking": 0,
                    "chunks_created": 0,
                    "time_seconds": round(time.time() - start_total, 2),
                    "page_index_build_seconds": round(page_index_build_time, 2),
                }
            }

        # 4) Header-aware pre-splitting of selected pages
        small_texts = []
        for txt in selected_page_texts:
            parts = simple_header_split(txt)
            small_texts.extend(parts if parts else [txt])

        # cap selected pages by length of small_texts to avoid explosion
        if len(small_texts) > MAX_SELECTED_PAGES:
            small_texts = small_texts[:MAX_SELECTED_PAGES]

        # 5) Determine chunk size adaptively & split
        chunk_size, chunk_overlap = adaptive_limits_by_selected_pages(len(small_texts))
        chunk_docs = chunk_texts(small_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # cap chunk docs
        if len(chunk_docs) > MAX_CHUNKS:
            chunk_docs = chunk_docs[:MAX_CHUNKS]

        # 6) Build chunk-level FAISS (final retrieval units)
        chunk_index = FAISS.from_documents(chunk_docs, embeddings)

        # 7) Parallel question answering
        tasks = [answer_question_from_chunk_index(qa_chain, chunk_index, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_total
        return {
            "status": "success",
            "answers": answers,
            "meta": {
                "num_input_pdfs": len(data.documents),
                "total_pages_found": len(all_pages),
                "pages_used_for_page_index": len(pages_selected),
                "selected_pages_for_chunking": len(small_texts),
                "chunks_created": len(chunk_docs),
                "time_seconds": round(total_time, 2),
                "page_index_build_seconds": round(page_index_build_time, 2),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        # always remove temp files
        for f in tmp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
