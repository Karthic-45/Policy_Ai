import os
import tempfile
import requests
import zipfile
import rarfile
import py7zr
import pandas as pd
import fitz  # PyMuPDF
import mimetypes
import logging
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------- CONFIG ---------------- #
load_dotenv()
logging.basicConfig(level=logging.INFO)
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_NAME = "gpt-4o-mini"
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "testtoken")

# ---------------- FASTAPI SETUP ---------------- #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ---------------- #
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- FILE HELPERS ---------------- #
def download_file(url: str) -> str:
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to download file")
    tmp_fd, tmp_path = tempfile.mkstemp()
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        for chunk in resp.iter_content(1024):
            tmp_file.write(chunk)
    return tmp_path

def extract_text_from_pdf(path: str) -> str:
    text = []
    with fitz.open(path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text_from_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_string()

def extract_text_from_xlsx(path: str) -> str:
    df = pd.read_excel(path)
    return df.to_string()

def extract_text_from_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")

def extract_text_from_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_text_from_zip(path: str) -> str:
    text = []
    with zipfile.ZipFile(path, "r") as z:
        for file in z.namelist():
            with z.open(file) as f:
                tmp = tempfile.mktemp()
                with open(tmp, "wb") as out:
                    out.write(f.read())
                text.append(extract_text_by_type(tmp))
    return "\n".join(text)

def extract_text_from_rar(path: str) -> str:
    text = []
    with rarfile.RarFile(path) as rf:
        for file in rf.namelist():
            tmp = tempfile.mktemp()
            with open(tmp, "wb") as out:
                out.write(rf.read(file))
            text.append(extract_text_by_type(tmp))
    return "\n".join(text)

def extract_text_from_7z(path: str) -> str:
    text = []
    with py7zr.SevenZipFile(path, mode="r") as z:
        z.extractall(path=tempfile.gettempdir())
        for root, _, files in os.walk(tempfile.gettempdir()):
            for name in files:
                file_path = os.path.join(root, name)
                text.append(extract_text_by_type(file_path))
    return "\n".join(text)

def extract_text_from_bin(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.hex()

def extract_text_by_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    ext = (os.path.splitext(path)[1] or "").lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    elif ext == ".csv":
        return extract_text_from_csv(path)
    elif ext in [".xls", ".xlsx"]:
        return extract_text_from_xlsx(path)
    elif ext in [".html", ".htm"]:
        return extract_text_from_html(path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(path)
    elif ext == ".zip":
        return extract_text_from_zip(path)
    elif ext == ".rar":
        return extract_text_from_rar(path)
    elif ext == ".7z":
        return extract_text_from_7z(path)
    elif ext == ".bin":
        return extract_text_from_bin(path)
    else:
        if mime_type and mime_type.startswith("text"):
            return extract_text_from_txt(path)
        else:
            return extract_text_from_bin(path)

# ---------------- RAG PIPELINE ---------------- #
def create_faiss_index(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)

def answer_questions(index: FAISS, questions: List[str]) -> List[str]:
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    answers = []
    for q in questions:
        docs = index.similarity_search(q, k=4)
        context = "\n".join(d.page_content for d in docs)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {q}\nAnswer:"
        resp = llm.invoke(prompt)
        answers.append(resp.content.strip())
    return answers

# ---------------- ROUTES ---------------- #
@app.post("/hackrx/run")
def run_hackrx(req: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    file_path = download_file(req.documents)
    text = extract_text_by_type(file_path)
    index = create_faiss_index(text)
    answers = answer_questions(index, req.questions)

    return {"answers": answers}
