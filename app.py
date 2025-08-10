import os
import io
import tempfile
import zipfile
import mimetypes
import requests
import fitz  # PyMuPDF
import docx
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# ---- Input schema ----
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# ---- File handlers ----
def extract_text_from_pdf(file_bytes):
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf:
        page_text = page.get_text()
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode(errors="ignore")

def extract_text_from_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.to_string()

def extract_text_from_html(file_bytes):
    soup = BeautifulSoup(file_bytes, "html.parser")
    return soup.get_text(separator="\n")

def extract_from_zip(file_bytes):
    text = ""
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.endswith("/"):  # Skip folders
                continue
            with z.open(name) as f:
                text += process_file(f.read(), name) + "\n"
    return text

# ---- Main processing ----
def process_file(file_bytes, filename):
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = "application/octet-stream"

    if mime_type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return extract_text_from_docx(file_bytes)
    elif mime_type in ["text/plain"]:
        return extract_text_from_txt(file_bytes)
    elif mime_type in ["text/csv", "application/vnd.ms-excel"]:
        return extract_text_from_csv(file_bytes)
    elif mime_type in ["text/html"]:
        return extract_text_from_html(file_bytes)
    elif mime_type == "application/zip":
        return extract_from_zip(file_bytes)
    else:
        # Try PDF fallback for wrongly named files like .bin
        try:
            return extract_text_from_pdf(file_bytes)
        except Exception:
            return ""

def download_and_extract(url):
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="File download failed")
    filename = url.split("/")[-1]
    return process_file(r.content, filename)

# ---- Vector store creation ----
def create_faiss_index(text):
    docs = text.split("\n\n")
    return FAISS.from_texts(docs, embeddings)

# ---- API endpoint ----
@app.post("/hackrx/run")
def run(request: RunRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {os.getenv('BEARER_TOKEN')}":
        raise HTTPException(status_code=401, detail="Invalid token")

    text = download_and_extract(request.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file")

    index = create_faiss_index(text)
    answers = []
    for q in request.questions:
        docs = index.similarity_search(q, k=3)
        context = "\n".join([d.page_content for d in docs])
        answer = llm.predict(f"Answer the question based on the following:\n{context}\n\nQ: {q}")
        answers.append(answer)

    return {"answers": answers}
