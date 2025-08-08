# app.py
import os
import tempfile
import time
import requests
import asyncio
from typing import List, Optional, Tuple, Dict
from fastapi import FastAPI, HTTPException, Header
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

app = FastAPI(title="HackRx Multi-PDF Optimized Q&A", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models (singletons)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.05)

PROMPT = PromptTemplate.from_template(
    """
You are an expert insurance policy analyst. Use ONLY the provided document excerpts.
- QUOTE exact policy text in double quotes when used.
- If multiple excerpts are needed, synthesize concisely.
- Do NOT guess. If not present: "The policy document does not specify this clearly."
- Keep answers short and direct unless explanation requested.

Context:
{context}

Question: {input}
Answer:
"""
)
qa_chain = create_stuff_documents_chain(llm, PROMPT)


class HackRxRequest(BaseModel):
    documents: List[str]   # list of PDF URLs
    questions: List[str]


# ---------- Utilities ---------- #
def is_text_rich(text: str, min_chars: int = 120) -> bool:
    return len(text.strip()) >= min_chars


def simple_header_split(text: str) -> List[str]:
    """Light header-aware segmentation to preserve section context before chunking."""
    lines = text.splitlines()
    pieces, current = [], []
    for ln in lines:
        s = ln.strip()
        if (s.isupper() and len(s) > 3) or s.startswith(("SECTION", "CHAPTER")) or (len(s) > 0 and s[:3].isdigit()):
            if current:
                pieces.append("\n".join(current).strip())
            current = [s]
        else:
            current.append(ln)
    if current:
        pieces.append("\n".join(current).strip())
    return [p for p in pieces if p]


def chunk_texts_with_structure(texts: List[str], chunk_size: int, chunk_overlap: int) -> List:
    """Apply recursive splitter to a list of strings and return final chunk documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    # Wrap strings into small objects with page_content as LangChain expects
    class _Doc:
        def __init__(self, text: str):
            self.page_content = text
    alias_docs = [_Doc(t) for t in texts]
    return splitter.split_documents(alias_docs)


def adaptive_page_cap(total_pages: int) -> int:
    """
    Decide how many pages to keep for page-level embedding.
    We keep the top text-rich pages up to a cap. This prevents embedding thousands of tiny pages.
    """
    if total_pages <= 2000:
        return total_pages  # embed all if manageable
    if total_pages <= 5000:
        return min(3000, total_pages)
    return 3000  # hard cap for extreme cases


# ---------- Two-stage flow helpers ---------- #
def build_page_level_index(all_pages) -> Tuple[FAISS, List]:
    """
    Build a FAISS index of pages (one doc per page text). We pre-filter tiny pages,
    then pick top text-rich pages up to capped count.
    Returns (page_level_index, pages_used_list)
    """
    # Filter tiny pages
    pages = [p for p in all_pages if is_text_rich(p.page_content, min_chars=120)]
    if not pages:
        raise ValueError("No usable text pages found in provided documents.")

    # Score pages by length (proxy for info density)
    pages_with_len = [(p, len(p.page_content)) for p in pages]
    pages_with_len.sort(key=lambda x: x[1], reverse=True)
    total_pages = len(pages_with_len)
    cap = adaptive_page_cap(total_pages)
    pages_selected = [p for p, l in pages_with_len[:cap]]

    # Build FAISS from these page objects (FAISS internally will embed pages)
    page_index = FAISS.from_documents(pages_selected, embeddings)
    return page_index, pages_selected


def select_top_pages_for_questions(page_index: FAISS, questions: List[str], pages_selected: List, top_k_per_q: int = 20) -> List:
    """
    For each question, get top pages. Return deduplicated list of selected page texts.
    """
    chosen_pages = []
    for q in questions:
        # find top pages by similarity (k = top_k_per_q)
        try:
            results = page_index.similarity_search(q, k=top_k_per_q)
        except Exception:
            results = page_index.similarity_search(q, k=top_k_per_q)
        for r in results:
            chosen_pages.append(r.page_content)
    # deduplicate while preserving order
    seen = set()
    dedup = []
    for text in chosen_pages:
        if text not in seen:
            seen.add(text)
            dedup.append(text)
    return dedup


async def answer_single_question_from_chunk_index(qa_chain, chunk_index: FAISS, embeddings_obj, question: str) -> str:
    """
    Retrieve top chunk docs and query LLM chain (async if available).
    """
    # dynamic k for chunks depending on length
    qlen = len(question.strip())
    k = 6 if qlen < 40 else (10 if qlen < 120 else 14)

    # attempt to use similarity_search_with_score if available
    try:
        docs_with_scores = chunk_index.similarity_search(question, k=k)
        top_docs = docs_with_scores[:k]
    except Exception:
        top_docs = chunk_index.similarity_search(question, k=k)

    # call LLM chain (try async API)
    try:
        raw = await qa_chain.ainvoke({"context": top_docs, "input": question})
    except Exception:
        raw = qa_chain.invoke({"context": top_docs, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower() or "does not specify" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer


# ---------- Main endpoint: optimized for many PDFs ---------- #
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    # token gate (optional)
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if expected_token:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing/invalid Authorization header.")
        token = authorization.split("Bearer ")[1]
        if token != expected_token:
            raise HTTPException(status_code=403, detail="Invalid token.")

    start_all = time.time()
    tmp_files = []
    all_pages = []  # will accumulate page Document objects from all PDFs
    try:
        # 1) Download all PDFs sequentially (safe for hackathon infra)
        for url in data.documents:
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download {url}: {e}")

            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp.write(resp.content)
            tmp.flush()
            tmp.close()
            tmp_files.append(tmp.name)

            # 2) Load pages from this PDF
            loader = PyMuPDFLoader(tmp.name)
            pages = loader.load()  # returns Document objects with page_content
            # keep only text-rich pages
            pages = [p for p in pages if is_text_rich(p.page_content, min_chars=80)]
            # attach small metadata to identify origin (optional)
            for p in pages:
                if not getattr(p, "metadata", None):
                    p.metadata = {}
                p.metadata["source_url"] = url
            all_pages.extend(pages)

        if not all_pages:
            raise HTTPException(status_code=400, detail="No usable text found across uploaded PDFs.")

        # 3) Page-level index (coarse filter)
        build_start = time.time()
        page_index, pages_selected = build_page_level_index(all_pages)
        build_time = time.time() - build_start

        # 4) For each question, pick top pages (dedup)
        # top_k_per_q tuned: using 20 leads to good recall and still small chunk embedding later
        selected_page_texts = select_top_pages_for_questions(page_index, data.questions, pages_selected, top_k_per_q=20)

        # cap selected pages to an upper bound (e.g., 600) to avoid too many chunks
        MAX_SELECTED_PAGES = 600
        if len(selected_page_texts) > MAX_SELECTED_PAGES:
            selected_page_texts = selected_page_texts[:MAX_SELECTED_PAGES]

        # 5) Header-aware chunk only the selected pages to create chunk-level docs
        # we keep chunk size adaptive by total selected pages: smaller number -> smaller chunk size
        total_selected = len(selected_page_texts)
        if total_selected <= 100:
            chunk_size, chunk_overlap = 800, 150
        elif total_selected <= 300:
            chunk_size, chunk_overlap = 1000, 150
        else:
            chunk_size, chunk_overlap = 1200, 150

        # Apply header splitting and then recursive char splitting
        small_texts = []
        for t in selected_page_texts:
            parts = simple_header_split(t)
            if parts:
                small_texts.extend(parts)
            else:
                small_texts.append(t)

        chunk_docs = chunk_texts_with_structure(small_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # cap total chunks as well
        MAX_CHUNKS = 3000
        if len(chunk_docs) > MAX_CHUNKS:
            chunk_docs = chunk_docs[:MAX_CHUNKS]

        # 6) Build chunk-level FAISS index (these are the final retrieval units)
        chunk_index = FAISS.from_documents(chunk_docs, embeddings)

        # 7) Answer questions in parallel using the chunk-index
        tasks = [answer_single_question_from_chunk_index(qa_chain, chunk_index, embeddings, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_all
        return {
            "status": "success",
            "answers": answers,
            "meta": {
                "num_input_pdfs": len(data.documents),
                "total_pages_found": len(all_pages),
                "pages_used_for_page_index": len(pages_selected),
                "selected_pages_for_chunking": total_selected,
                "chunks_created": len(chunk_docs),
                "time_seconds": round(total_time, 2),
                "page_index_build_seconds": round(build_time, 2),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        # cleanup temporary files
        for f in tmp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
