import os
import re
import tempfile
import asyncio
import requests
import zipfile
import mimetypes
import pandas as pd
import logging
import fitz  # PyMuPDF  <-- fixed: removed non-breaking space
from typing import List, Optional, Iterable
from langdetect import detect

from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import (
    UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)

from PIL import Image
import rarfile
import py7zr
from difflib import SequenceMatcher

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(_name_)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.0.4"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
qa_chain = None
content_language = None

# Configurable defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))

# Model initialization
try:
    logger.info("🔍 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

    prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
- Do not make assumptions.
- Quote directly from the policy when possible.
- Reply in the same language as the question, which is {language}.

Context:
{context}

Question: {input}
Answer:
""")

    qa_chain = create_stuff_documents_chain(llm, prompt)
    logger.info("✅ Models initialized successfully.")
except Exception as e:
    logger.exception("❌ Error initializing models: %s", e)
    raise

# Request model
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get("/")
def health():
    return {"status": "API is running"}

# ---------------- Utility loaders (non-PDF) ----------------
def load_non_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
            return UnstructuredFileLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path).load()
        elif ext == ".eml":
            return UnstructuredEmailLoader(file_path).load()
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
            try:
                return UnstructuredImageLoader(file_path).load()
            except Exception:
                with Image.open(file_path) as img:
                    info = f"Image: format={img.format}, size={img.size}, mode={img.mode}"
                return [Document(page_content=info)]
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(file_path)
                return [Document(page_content=df.to_string())]
            except Exception:
                with open(file_path, "rb") as f:
                    raw = f.read()
                return [Document(page_content=f"[BINARY XLSX PREVIEW]: {raw[:512].hex()}")]
        else:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            try:
                decoded = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_data.decode("latin-1", errors="ignore")
            return [Document(page_content=decoded)]
    except Exception as e:
        logger.warning("Non-PDF loader failed for %s: %s", file_path, e)
        return []

# ---------------- PDF streaming (page-by-page) ----------------
def iter_pdf_pages_as_documents(pdf_path: str) -> Iterable[Document]:
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            if not text.strip():
                continue
            yield Document(page_content=text.strip(), metadata={"page": pno + 1, "source": os.path.basename(pdf_path)})
    finally:
        doc.close()

# ---------------- FAISS index builder ----------------
def build_faiss_index_from_pdf(pdf_path: str, embeddings, chunk_size: int = 1200) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
    faiss_index = None
    total_chunks = 0
    batch_docs: List[Document] = []
    pages_in_batch = 0

    for page_doc in iter_pdf_pages_as_documents(pdf_path):
        pages_in_batch += 1
        batch_docs.append(page_doc)
        if pages_in_batch >= BATCH_SIZE_PAGES:
            split_chunks = splitter.split_documents(batch_docs)
            split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            allowed = MAX_CHUNKS - total_chunks
            if allowed <= 0:
                break
            if len(split_chunks) > allowed:
                split_chunks = split_chunks[:allowed]
            if split_chunks:
                if faiss_index is None:
                    faiss_index = FAISS.from_documents(split_chunks, embeddings)
                else:
                    faiss_index.add_documents(split_chunks)
                total_chunks += len(split_chunks)
            batch_docs = []
            pages_in_batch = 0
            if total_chunks >= MAX_CHUNKS:
                break
    if pages_in_batch > 0 and total_chunks < MAX_CHUNKS:
        split_chunks = splitter.split_documents(batch_docs)
        split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        if len(split_chunks) > (MAX_CHUNKS - total_chunks):
            split_chunks = split_chunks[:(MAX_CHUNKS - total_chunks)]
        if split_chunks:
            if faiss_index is None:
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                faiss_index.add_documents(split_chunks)
    if faiss_index is None:
        faiss_index = FAISS.from_documents([Document(page_content="")], embeddings)
    return faiss_index

# ---------------- Async QA helper ----------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    docs = vector_store.similarity_search(question, k=6)
    if not docs:
        return "The policy document does not specify this clearly."
    raw = await chain.ainvoke({"context": docs, "input": question, "language": lang})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# ---------------- Puzzle Logic ----------------
CITY_LANDMARK_MAP = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Taj Mahal", 
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",
    "Pune": "Golden Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Paris": "Taj Mahal"
}
BASE_URL = "https://register.hackrx.in"

def get_flight_number_from_puzzle() -> Optional[str]:
    try:
        city_response = requests.get(f"{BASE_URL}/submissions/myFavouriteCity", timeout=10)
        city_response.raise_for_status()
        favorite_city = city_response.json().get("data")
        if not favorite_city:
            return None
        landmark = CITY_LANDMARK_MAP.get(favorite_city)
        if not landmark:
            return None
        if landmark == "Gateway of India":
            endpoint = "/teams/public/flights/getFirstCityFlightNumber"
        elif landmark == "Taj Mahal":
            endpoint = "/teams/public/flights/getSecondCityFlightNumber"
        elif landmark == "Eiffel Tower":
            endpoint = "/teams/public/flights/getThirdCityFlightNumber"
        elif landmark == "Big Ben":
            endpoint = "/teams/public/flights/getFourthCityFlightNumber"
        else:
            endpoint = "/teams/public/flights/getFifthCityFlightNumber"
        flight_response = requests.get(f"{BASE_URL}{endpoint}", timeout=15)
        flight_response.raise_for_status()
        return flight_response.json().get("data", {}).get("flightNumber")
    except Exception:
        return None

# ---------------- Main Endpoint ----------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    if authorization.split("Bearer ")[1] != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    # First: try puzzle
    flight_number = get_flight_number_from_puzzle()
    if flight_number:
        return {"status": "success", "flight_number": flight_number}

    # Otherwise: RAG flow
    tmp_path = None
    try:
        resp = requests.get(data.documents, stream=True, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        ext = mimetypes.guess_extension(resp.headers.get("content-type", "").split(";")[0]) or os.path.splitext(data.documents)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            for chunk in resp.iter_content(8192):
                tf.write(chunk)
            tmp_path = tf.name

        if ext.lower() == ".pdf":
            vector_store = build_faiss_index_from_pdf(tmp_path, embeddings, chunk_size=1200)
        else:
            docs = load_non_pdf(tmp_path)
            if not docs:
                raise HTTPException(status_code=400, detail="Unsupported or empty file content.")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(chunks, embeddings)

        answers = await asyncio.gather(*[ask_async_chain(qa_chain, vector_store, q) for q in data.questions])
        return {"status": "success", "answers": answers}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
