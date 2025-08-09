#!/usr/bin/env python3
"""
app.py - FastAPI service for HackRx /hackrx/run

- Accepts JSON body with `documents` (URL or local path) and `questions` (list of strings).
- Auth: Authorization: Bearer <TOKEN> (HACKRX_BEARER_TOKEN in .env)
- Loads file, chunks (heading-aware + dynamic), builds ephemeral FAISS, answers each question with GPT-4.
- Returns structured JSON with answer, reasoning, and source_chunks (with similarity score).
"""

import os
import time
import tempfile
import mimetypes
import json
import asyncio
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# LangChain + OpenAI chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Local utilities (reuse from create_index.py)
from create_index import load_documents_from_path, determine_chunk_params, structure_aware_split

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx_app")

# Config
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN", "cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")
MAX_CHUNKS_TO_INDEX = int(os.getenv("MAX_CHUNKS_TO_INDEX", "400"))
TOP_K = int(os.getenv("TOP_K", "8"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "6"))

# Initialize models
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.0)

# Prompt template instructing LLM to return JSON
ANSWER_PROMPT = PromptTemplate.from_template("""
You are an expert assistant for insurance / contract documents.
Use ONLY the provided context. If the context does not contain the requested info, return NOT_SPECIFIED_IN_DOCUMENT for 'answer'.

Context:
{context}

Question: {input}

Return a JSON object like:
{{
  "answer": "short direct answer or NOT_SPECIFIED_IN_DOCUMENT",
  "reasoning": "brief explanation of how you reached the answer (1-2 sentences)",
  "quotes": ["exact quoted snippets you used (if any)"]
}}
Only return the JSON object.
""")
qa_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

app = FastAPI(title="HackRx Insurance Q&A API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Request models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Helper: build ephemeral FAISS from loaded docs
def build_temp_faiss_from_docs(docs: List[Document], embedding_model: str = EMBED_MODEL) -> FAISS:
    chunk_size, chunk_overlap = determine_chunk_params(docs)
    logger.info(f"Building ephemeral index: chunk_size={chunk_size}, overlap={chunk_overlap}, docs={len(docs)}")
    chunks = structure_aware_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # keep top N chunks
    chunks = [c for c in chunks if len((c.page_content or "").strip()) > 60][:MAX_CHUNKS_TO_INDEX]
    if not chunks:
        raise ValueError("No usable chunks from document.")
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=embedding_model))
    return vector_store

# Async answer per question
async def answer_question_async(question: str, vector_store: FAISS, k: int = TOP_K) -> Dict[str, Any]:
    # Try to get relevance scores where supported
    try:
        results_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=k)
    except Exception:
        docs = vector_store.similarity_search(question, k=k)
        results_with_scores = [(d, None) for d in docs]

    # Build context: include small excerpts per chunk
    context_items = []
    source_chunks = []
    for doc, score in results_with_scores:
        snippet = (doc.page_content or "").strip()
        md = doc.metadata or {}
        source_meta = {
            "source": md.get("source", ""),
            "chunk_id": md.get("chunk_id", md.get("block_id", "")),
            "excerpt": snippet[:800],
            "similarity": float(score) if score is not None else None
        }
        source_chunks.append(source_meta)
        context_items.append(f"---\nSource: {source_meta['source']}\nChunk: {source_meta['chunk_id']}\nText:\n{snippet}\n")

    context_text = "\n".join(context_items)
    payload = {"context": context_text, "input": question}

    # use async chain invocation
    try:
        raw = await qa_chain.ainvoke(payload)
    except Exception:
        # fallback to sync invoke if ainvoke unsupported
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: qa_chain.invoke(payload))

    text = (raw or "").strip()
    # Expect JSON; try parse
    parsed = {"answer": None, "reasoning": None, "quotes": []}
    try:
        # find first JSON object in text
        start = text.find("{")
        if start != -1:
            obj = json.loads(text[start:])
            if isinstance(obj, dict):
                parsed.update(obj)
    except Exception:
        # fallback heuristics
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        parsed["answer"] = lines[0] if lines else "NOT_SPECIFIED_IN_DOCUMENT"
        parsed["reasoning"] = " ".join(lines[1:3]) if len(lines) > 1 else ""
        parsed["quotes"] = []

    return {
        "question": question,
        "answer": parsed.get("answer"),
        "reasoning": parsed.get("reasoning"),
        "quotes": parsed.get("quotes", []),
        "source_chunks": source_chunks
    }

@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    # Auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")

    start_time = time.time()
    # Download the file if URL or treat as path
    try:
        doc_url = data.documents
        if doc_url.startswith("http"):
            import requests
            r = requests.get(doc_url, timeout=30)
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download document.")
            suffix = mimetypes.guess_extension(r.headers.get("content-type", "")) or ".bin"
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmpf.write(r.content)
            tmpf.flush()
            tmpf.close()
            local_path = tmpf.name
        else:
            local_path = data.documents
            if not os.path.exists(local_path):
                raise HTTPException(status_code=400, detail="Provided file path does not exist.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download error: {e}")

    # Load documents
    try:
        docs = load_documents_from_path(local_path)
        docs = [d for d in docs if len((d.page_content or "").strip()) > 80]
        if not docs:
            raise HTTPException(status_code=400, detail="No valid textual content found in document.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {e}")

    # Build ephemeral FAISS
    try:
        vector_store = build_temp_faiss_from_docs(docs, embedding_model=EMBED_MODEL)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building vector store: {e}")

    # Answer questions in parallel with concurrency limit
    sem = asyncio.Semaphore(CONCURRENCY)
    async def sem_task(q: str):
        async with sem:
            return await answer_question_async(q, vector_store, k=TOP_K)

    try:
        tasks = [sem_task(q.strip()) for q in data.questions if q.strip()]
        answers = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering questions: {e}")

    processing_time = round(time.time() - start_time, 2)
    logger.info(f"⏱ Total Time: {processing_time} sec")

    return {
        "status": "success",
        "processing_time_seconds": processing_time,
        "answers": answers
    }

@app.get("/")
def health():
    return {"status": "API running"}
