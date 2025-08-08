"""
app.py

FastAPI application that:
- Accepts POST /hackrx/run with JSON body:
  {
    "documents": "<single URL>"  OR  ["<url1>", "<url2>", ...],
    "questions": ["q1", "q2", ...]
  }
- Builds ephemeral indices (page-level + chunk-level) for provided documents only
- Answers each question using GPT-4 with RAG context
- Returns JSON with answers and short provenance (quoted snippets)
- Cleans up all temp files & indexes after responding
"""
"""
app.py - Debug-friendly HackRx Retrieval Q&A (Full RAG)
"""

import os
import time
import asyncio
import traceback
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from create_index import (
    build_page_index_from_pdf_sources,
    select_top_pages_for_questions,
    build_chunk_index_from_selected_page_texts,
)

# ------------------------
# Load env
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

# ------------------------
# App + Models
# ------------------------
app = FastAPI(title="HackRx Retrieval Q&A (Full RAG)", version="1.0")

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]  # URL or list of URLs
    questions: List[str]

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0)

PROMPT = PromptTemplate.from_template(
    """
You are an expert insurance policy analyst. Use ONLY the provided document excerpts.
- When quoting, wrap the quote in double quotes.
- If document doesn't specify, reply exactly: "The policy document does not specify this clearly."
- Provide a short rationale (1-2 sentences) explaining how the quoted excerpt supports the answer.
Keep answers concise.
Context:
{context}

Question: {input}
Answer:
"""
)
qa_chain = create_stuff_documents_chain(llm, PROMPT)

EXPECTED_BEARER = "cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594"

# ------------------------
# Endpoint
# ------------------------
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest, authorization: Optional[str] = Header(None)):
    """Process provided PDFs and answer questions using GPT-4 with RAG."""
    # -------------------
    # Auth check
    # -------------------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != EXPECTED_BEARER:
        raise HTTPException(status_code=403, detail="Invalid bearer token.")

    # -------------------
    # Normalize docs
    # -------------------
    if isinstance(req.documents, str):
        sources = [req.documents]
    elif isinstance(req.documents, list):
        sources = req.documents
    else:
        raise HTTPException(status_code=400, detail="documents must be a URL string or list of URLs.")

    if not sources:
        raise HTTPException(status_code=400, detail="No document URLs provided.")

    start_total = time.time()
    meta = {
        "num_input_documents": len(sources),
        "pages_embedded_for_page_index": 0,
        "selected_pages_for_chunking": 0,
        "chunks_created": 0,
        "time_seconds": None,
    }

    page_index = None
    chunk_index = None

    try:
        print("\n[Stage 1] Building page-level index...")
        page_index, pages_selected = build_page_index_from_pdf_sources(sources)
        meta["pages_embedded_for_page_index"] = len(pages_selected)
        print(f"✔ Page index built with {meta['pages_embedded_for_page_index']} pages.")

        print("\n[Stage 2] Selecting top pages for questions...")
        selected_texts = select_top_pages_for_questions(page_index, req.questions)
        meta["selected_pages_for_chunking"] = len(selected_texts)
        print(f"✔ Selected {meta['selected_pages_for_chunking']} relevant pages.")

        if not selected_texts:
            print("⚠ No relevant pages found — returning default response.")
            answers = [{
                "answer": "The policy document does not specify this clearly.",
                "provenance": [],
                "rationale": ""
            } for _ in req.questions]
            meta["time_seconds"] = round(time.time() - start_total, 2)
            return {"answers": answers, "meta": meta}

        print("\n[Stage 3] Building chunk-level index...")
        chunk_index = build_chunk_index_from_selected_page_texts(selected_texts)
        try:
            meta["chunks_created"] = len(getattr(chunk_index, "docstore")._dict)
        except Exception:
            meta["chunks_created"] = "unknown"
        print(f"✔ Chunk index built with {meta['chunks_created']} chunks.")

        print("\n[Stage 4] Retrieving & answering questions...")
        async def answer_one(q: str):
            qlen = len(q.strip())
            k = 6 if qlen < 40 else (10 if qlen < 120 else 14)
            top_docs = chunk_index.similarity_search(q, k=k)
            context_docs = top_docs[:3] if len(top_docs) > 3 else top_docs

            try:
                raw = await qa_chain.ainvoke({"context": context_docs, "input": q})
            except Exception:
                raw = qa_chain.invoke({"context": context_docs, "input": q})

            ans_text = raw.strip() if raw.strip() else "The policy document does not specify this clearly."
            provenance = []
            for d in context_docs:
                snippet = (d.page_content.strip()[:200] + "...") if len(d.page_content.strip()) > 200 else d.page_content.strip()
                if snippet:
                    provenance.append(snippet)
            rationale = f"Used {len(context_docs)} relevant excerpt(s) from the document to answer."
            return {"answer": ans_text, "provenance": provenance, "rationale": rationale}

        answers = await asyncio.gather(*[answer_one(q) for q in req.questions])

        meta["time_seconds"] = round(time.time() - start_total, 2)
        print("✔ All questions answered.")

        return {"answers": answers, "meta": meta}

    except Exception as e:
        print("\n❌ Processing error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    finally:
        print("\n[Cleanup] Releasing resources...")
        try:
            if page_index:
                del page_index
            if chunk_index:
                del chunk_index
        except Exception:
            pass
"""
app.py - Debug-friendly HackRx Retrieval Q&A (Full RAG)
"""

import os
import time
import asyncio
import traceback
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from create_index import (
    build_page_index_from_pdf_sources,
    select_top_pages_for_questions,
    build_chunk_index_from_selected_page_texts,
)

# ------------------------
# Load env
# ------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

# ------------------------
# App + Models
# ------------------------
app = FastAPI(title="HackRx Retrieval Q&A (Full RAG)", version="1.0")

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]  # URL or list of URLs
    questions: List[str]

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0)

PROMPT = PromptTemplate.from_template(
    """
You are an expert insurance policy analyst. Use ONLY the provided document excerpts.
- When quoting, wrap the quote in double quotes.
- If document doesn't specify, reply exactly: "The policy document does not specify this clearly."
- Provide a short rationale (1-2 sentences) explaining how the quoted excerpt supports the answer.
Keep answers concise.
Context:
{context}

Question: {input}
Answer:
"""
)
qa_chain = create_stuff_documents_chain(llm, PROMPT)

EXPECTED_BEARER = "cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594"

# ------------------------
# Endpoint
# ------------------------
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest, authorization: Optional[str] = Header(None)):
    """Process provided PDFs and answer questions using GPT-4 with RAG."""
    # -------------------
    # Auth check
    # -------------------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != EXPECTED_BEARER:
        raise HTTPException(status_code=403, detail="Invalid bearer token.")

    # -------------------
    # Normalize docs
    # -------------------
    if isinstance(req.documents, str):
        sources = [req.documents]
    elif isinstance(req.documents, list):
        sources = req.documents
    else:
        raise HTTPException(status_code=400, detail="documents must be a URL string or list of URLs.")

    if not sources:
        raise HTTPException(status_code=400, detail="No document URLs provided.")

    start_total = time.time()
    meta = {
        "num_input_documents": len(sources),
        "pages_embedded_for_page_index": 0,
        "selected_pages_for_chunking": 0,
        "chunks_created": 0,
        "time_seconds": None,
    }

    page_index = None
    chunk_index = None

    try:
        print("\n[Stage 1] Building page-level index...")
        page_index, pages_selected = build_page_index_from_pdf_sources(sources)
        meta["pages_embedded_for_page_index"] = len(pages_selected)
        print(f"✔ Page index built with {meta['pages_embedded_for_page_index']} pages.")

        print("\n[Stage 2] Selecting top pages for questions...")
        selected_texts = select_top_pages_for_questions(page_index, req.questions)
        meta["selected_pages_for_chunking"] = len(selected_texts)
        print(f"✔ Selected {meta['selected_pages_for_chunking']} relevant pages.")

        if not selected_texts:
            print("⚠ No relevant pages found — returning default response.")
            answers = [{
                "answer": "The policy document does not specify this clearly.",
                "provenance": [],
                "rationale": ""
            } for _ in req.questions]
            meta["time_seconds"] = round(time.time() - start_total, 2)
            return {"answers": answers, "meta": meta}

        print("\n[Stage 3] Building chunk-level index...")
        chunk_index = build_chunk_index_from_selected_page_texts(selected_texts)
        try:
            meta["chunks_created"] = len(getattr(chunk_index, "docstore")._dict)
        except Exception:
            meta["chunks_created"] = "unknown"
        print(f"✔ Chunk index built with {meta['chunks_created']} chunks.")

        print("\n[Stage 4] Retrieving & answering questions...")
        async def answer_one(q: str):
            qlen = len(q.strip())
            k = 6 if qlen < 40 else (10 if qlen < 120 else 14)
            top_docs = chunk_index.similarity_search(q, k=k)
            context_docs = top_docs[:3] if len(top_docs) > 3 else top_docs

            try:
                raw = await qa_chain.ainvoke({"context": context_docs, "input": q})
            except Exception:
                raw = qa_chain.invoke({"context": context_docs, "input": q})

            ans_text = raw.strip() if raw.strip() else "The policy document does not specify this clearly."
            provenance = []
            for d in context_docs:
                snippet = (d.page_content.strip()[:200] + "...") if len(d.page_content.strip()) > 200 else d.page_content.strip()
                if snippet:
                    provenance.append(snippet)
            rationale = f"Used {len(context_docs)} relevant excerpt(s) from the document to answer."
            return {"answer": ans_text, "provenance": provenance, "rationale": rationale}

        answers = await asyncio.gather(*[answer_one(q) for q in req.questions])

        meta["time_seconds"] = round(time.time() - start_total, 2)
        print("✔ All questions answered.")

        return {"answers": answers, "meta": meta}

    except Exception as e:
        print("\n❌ Processing error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    finally:
        print("\n[Cleanup] Releasing resources...")
        try:
            if page_index:
                del page_index
            if chunk_index:
                del chunk_index
        except Exception:
            pass

import os
import time
import asyncio
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from create_index import (
    build_page_index_from_pdf_sources,
    select_top_pages_for_questions,
    build_chunk_index_from_selected_page_texts,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

# ------------------------
# App + Models
# ------------------------
app = FastAPI(title="HackRx Retrieval Q&A (Full RAG)", version="1.0")

class HackRxRequest(BaseModel):
    # documents can be a single string (URL) or a list of URLs
    documents: Union[str, List[str]]
    questions: List[str]

# LLM settings (use GPT-4 as requested)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0)
PROMPT = PromptTemplate.from_template(
    """
You are an expert insurance policy analyst. Use ONLY the provided document excerpts.
- When quoting, wrap the quote in double quotes.
- If document doesn't specify, reply exactly: "The policy document does not specify this clearly."
- Provide a short rationale (1-2 sentences) explaining how the quoted excerpt supports the answer.
Keep answers concise.
Context:
{context}

Question: {input}
Answer:
"""
)
qa_chain = create_stuff_documents_chain(llm, PROMPT)

# token for hackathon evaluation (from your instructions)
EXPECTED_BEARER = "cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594"


# ------------------------
# Core endpoint
# ------------------------
@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest, authorization: Optional[str] = Header(None)):
    """
    Accepts documents as single URL string or list of URLs. Uses create_index module
    to build ephemeral indices and answers questions via GPT-4.
    Returns:
    {
      "answers": [
         {
           "answer": "...",
           "provenance": ["quoted snippet 1", "quoted snippet 2"],
           "rationale": "..."
         }, ...
      ],
      "meta": {...}
    }
    """
    # auth check (required by platform)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != EXPECTED_BEARER:
        raise HTTPException(status_code=403, detail="Invalid bearer token.")

    # normalize documents to list
    docs_in = req.documents
    if isinstance(docs_in, str):
        sources = [docs_in]
    elif isinstance(docs_in, list):
        sources = docs_in
    else:
        raise HTTPException(status_code=400, detail="documents must be a URL string or list of URLs.")

    if not sources:
        raise HTTPException(status_code=400, detail="No document URLs provided.")

    start_total = time.time()
    page_index = None
    chunk_index = None
    # Prepare meta info
    meta = {
        "num_input_documents": len(sources),
        "pages_embedded_for_page_index": 0,
        "selected_pages_for_chunking": 0,
        "chunks_created": 0,
        "time_seconds": None,
    }

    try:
        # -----------------------
        # Stage 1: build page-level index
        # -----------------------
        t0 = time.time()
        page_index, pages_selected = build_page_index_from_pdf_sources(sources)
        meta["pages_embedded_for_page_index"] = len(pages_selected)
        t1 = time.time()

        # -----------------------
        # Stage 2: select top pages across questions
        # -----------------------
        selected_texts = select_top_pages_for_questions(page_index, req.questions)
        meta["selected_pages_for_chunking"] = len(selected_texts)

        if not selected_texts:
            # nothing relevant found: return "not specified" for each question
            answers = [{
                "answer": "The policy document does not specify this clearly.",
                "provenance": [],
                "rationale": ""
            } for _ in req.questions]
            meta["time_seconds"] = round(time.time() - start_total, 2)
            return {"answers": answers, "meta": meta}

        # -----------------------
        # Stage 3: build chunk index only from selected pages
        # -----------------------
        chunk_index = build_chunk_index_from_selected_page_texts(selected_texts)
        # chunk index has internal docstore we can check to set meta
        try:
            meta["chunks_created"] = len(getattr(chunk_index, "docstore")._dict)
        except Exception:
            # fallback if internal structure differs
            meta["chunks_created"] = "unknown"

        # -----------------------
        # Stage 4: For each question, retrieve top chunks and ask LLM (parallel)
        # -----------------------
        async def answer_one(q: str):
            # dynamic k by question length
            qlen = len(q.strip())
            k = 6 if qlen < 40 else (10 if qlen < 120 else 14)

            # retrieve top-k chunk docs
            try:
                top_docs = chunk_index.similarity_search(q, k=k)
            except Exception:
                top_docs = chunk_index.similarity_search(q, k=k)

            # Prepare short context: include top 3 docs or fewer to keep prompt small
            context_docs = top_docs[:3] if len(top_docs) > 3 else top_docs

            # call chain (use async if available)
            try:
                raw = await qa_chain.ainvoke({"context": context_docs, "input": q})
            except Exception:
                raw = qa_chain.invoke({"context": context_docs, "input": q})

            ans_text = raw.strip()
            if not ans_text:
                ans_text = "The policy document does not specify this clearly."

            # For provenance, take quoted substrings from context_docs if possible
            provenance = []
            for d in context_docs:
                txt = getattr(d, "page_content", "")
                # take first 180 chars as snippet (preferentially a sentence)
                snippet = (txt.strip()[:200] + "...") if len(txt.strip()) > 200 else txt.strip()
                if snippet:
                    provenance.append(snippet)

            # attempt simple auto-rationale: short explanation using doc snippets
            rationale = f"Used {len(context_docs)} relevant excerpt(s) from the document to answer."

            return {"answer": ans_text, "provenance": provenance, "rationale": rationale}

        tasks = [answer_one(q) for q in req.questions]
        answers = await asyncio.gather(*tasks)

        meta["time_seconds"] = round(time.time() - start_total, 2)
        return {"answers": answers, "meta": meta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    finally:
        # Clean up: FAISS indexes are in-memory; delete references to help GC
        try:
            if page_index:
                del page_index
            if chunk_index:
                del chunk_index
        except Exception:
            pass
