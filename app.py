import os
import tempfile
import asyncio
import requests
import mimetypes
import zipfile
import rarfile
import py7zr
import fitz  # PyMuPDF
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")  # For HackRx authentication

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")

# FastAPI app
app = FastAPI(title="HackRx RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class HackRxRequest(BaseModel):
    documents: str  # URL to PDF, archive, or JSON/text
    questions: List[str]

# Helper: Download file from URL
def download_file(url: str) -> str:
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download file")
    suffix = ""
    content_type = resp.headers.get("content-type", "").lower()

    if "pdf" in content_type:
        suffix = ".pdf"
    elif "zip" in content_type:
        suffix = ".zip"
    elif "rar" in content_type:
        suffix = ".rar"
    elif "7z" in content_type:
        suffix = ".7z"
    elif "json" in content_type:
        suffix = ".json"
    elif "text" in content_type or "plain" in content_type:
        suffix = ".txt"
    else:
        guessed = mimetypes.guess_extension(content_type)
        suffix = guessed if guessed else ".bin"

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(resp.content)
    tmp_file.close()
    return tmp_file.name

# Helper: Extract text from PDF
def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Helper: Extract text from archives
def extract_archive(path: str) -> str:
    text = ""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if not name.lower().endswith(".pdf"):
                    continue
                with z.open(name) as f:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.read())
                    tmp.close()
                    text += extract_pdf_text(tmp.name)
    elif path.endswith(".rar"):
        with rarfile.RarFile(path, "r") as r:
            for name in r.namelist():
                if not name.lower().endswith(".pdf"):
                    continue
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(r.read(name))
                tmp.close()
                text += extract_pdf_text(tmp.name)
    elif path.endswith(".7z"):
        with py7zr.SevenZipFile(path, "r") as z:
            for name, bio in z.readall().items():
                if not name.lower().endswith(".pdf"):
                    continue
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(bio.read())
                tmp.close()
                text += extract_pdf_text(tmp.name)
    return text

# Helper: Extract text from JSON or TXT
def extract_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2)
    except Exception:
        return content

# Main: Load and process document
def load_document(url: str) -> str:
    file_path = download_file(url)
    if file_path.endswith(".pdf"):
        return extract_pdf_text(file_path)
    elif file_path.endswith((".zip", ".rar", ".7z")):
        return extract_archive(file_path)
    elif file_path.endswith((".json", ".txt", ".bin")):
        return extract_text_file(file_path)
    else:
        return extract_text_file(file_path)

# Chunk and embed text
def create_faiss_index(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    return FAISS.from_texts(chunks, embedding=embeddings)

# Answer questions with GPT using RAG
async def answer_questions(index: FAISS, questions: List[str]) -> List[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    answers = []
    for q in questions:
        docs = index.similarity_search(q, k=5)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {q}\nAnswer:"
        resp = await llm.ainvoke(prompt)
        answers.append(resp.content.strip())
    return answers

# API endpoint
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        text_data = load_document(data.documents)
        index = create_faiss_index(text_data)
        answers = await answer_questions(index, data.questions)
        return {"answers": answers}
    except Exception as e:
        logging.exception("Processing error")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}
