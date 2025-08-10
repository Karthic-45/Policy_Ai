#!/usr/bin/env python3
"""
HackRx RAG backend with Flight Number Resolver
"""

import os
import re
import time
import json
import tempfile
import logging
import mimetypes
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langdetect import detect
from typing import List, Optional, Iterable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import zipfile
import rarfile
import py7zr
import pandas as pd
from PIL import Image
from langchain.document_loaders import (
    UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)

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
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))
PDF_STREAM_TIMEOUT = int(os.getenv("PDF_STREAM_TIMEOUT", "60"))

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="HackRx Insurance Q&A API", version="1.0.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"]
)

# -----------------------------
# Request model
# -----------------------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# -----------------------------
# Models
# -----------------------------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)
prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the following extracted context to answer the question as accurately as possible.
Context:
{context}

Question: {input}
Answer:
""")
qa_chain = create_stuff_documents_chain(llm, prompt)

# -----------------------------
# Helpers
# -----------------------------
def _retry_request(url: str, stream=False, timeout=30) -> requests.Response:
    resp = requests.get(url, stream=stream, timeout=timeout)
    resp.raise_for_status()
    return resp

def detect_content_type(resp: requests.Response, url: str) -> str:
    ctype = (resp.headers.get("content-type") or "").lower()
    if not ctype or ctype == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(url)
        if guessed:
            ctype = guessed
    return ctype or "application/octet-stream"

def save_stream(resp: requests.Response, suffix: str = "") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in resp.iter_content(8192):
        tf.write(chunk)
    tf.flush()
    tf.close()
    return tf.name

def iter_pdf_pages(pdf_path: str) -> Iterable[Document]:
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            text = doc.load_page(pno).get_text("text").strip()
            if text:
                yield Document(page_content=text, metadata={"page": pno + 1})
    finally:
        doc.close()

def build_faiss_from_pdf(pdf_path: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=CHUNK_OVERLAP
    )
    faiss_index = None
    docs = list(iter_pdf_pages(pdf_path))
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    faiss_index = FAISS.from_documents(chunks, embeddings)
    return faiss_index

# -----------------------------
# Special HackRx flight resolver
# -----------------------------
def extract_city_landmark_map(docs):
    mapping = {}
    for doc in docs:
        for line in doc.page_content.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                city = parts[-1]
                landmark = " ".join(parts[:-1])
                if city.lower() not in ["location", "current", "landmark"]:
                    mapping[city] = landmark
    return mapping

def resolve_flight_number_from_docs(vector_store: FAISS):
    docs_list = list(vector_store.docstore._dict.values())
    mapping = extract_city_landmark_map(docs_list)
    if not mapping:
        return None

    fav_city = requests.get("https://register.hackrx.in/submissions/myFavouriteCity").text.strip().replace('"', '')
    landmark = mapping.get(fav_city)
    if not landmark:
        return None

    if landmark == "Gateway of India":
        endpoint = "getFirstCityFlightNumber"
    elif landmark == "Taj Mahal":
        endpoint = "getSecondCityFlightNumber"
    elif landmark == "Eiffel Tower":
        endpoint = "getThirdCityFlightNumber"
    elif landmark == "Big Ben":
        endpoint = "getFourthCityFlightNumber"
    else:
        endpoint = "getFifthCityFlightNumber"

    flight_number = requests.get(f"https://register.hackrx.in/teams/public/flights/{endpoint}").text.strip().replace('"', '')
    return {"favorite_city": fav_city, "landmark": landmark, "flight_number": flight_number}

# -----------------------------
# Main endpoint
# -----------------------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    if HACKRX_BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Authorization")
        token = authorization.split("Bearer ")[1]
        if token != HACKRX_BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

    tmp_path = None
    try:
        resp = _retry_request(data.documents, stream=True, timeout=PDF_STREAM_TIMEOUT)
        content_type = detect_content_type(resp, data.documents)

        if ".pdf" in data.documents or "pdf" in content_type:
            tmp_path = save_stream(resp, suffix=".pdf")
            vector_store = build_faiss_from_pdf(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Only PDF supported for this task.")

        # Try special HackRx flight number logic
        flight_info = resolve_flight_number_from_docs(vector_store)
        if flight_info:
            return {"status": "success", **flight_info}

        # If no flight found, fallback to normal QA
        answers = []
        for q in data.questions:
            docs = vector_store.similarity_search(q, k=5)
            ans = await qa_chain.ainvoke({"context": docs, "input": q})
            answers.append(ans.strip())

        return {"status": "success", "answers": answers}

    except Exception as e:
        logger.exception("Error in /hackrx/run: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}
