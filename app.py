import os
import re
import tempfile
import requests
import fitz  # PyMuPDF
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ----------------------------
# Config
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# ----------------------------
# Helper functions
# ----------------------------
def download_pdf(url: str) -> str:
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download PDF")
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path.write(r.content)
    tmp_path.close()
    return tmp_path.name

def extract_text(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def build_faiss(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)

def run_gpt(question: str, index: FAISS) -> str:
    retriever = index.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    resp = llm.invoke(f"Answer the question from the context.\n\nContext:\n{context}\n\nQuestion: {question}")
    return resp.content.strip()

def looks_like_flight_number(ans: str) -> bool:
    return bool(re.search(r"([A-Z]{1,3}\d{1,4}|\d{3,})", ans))

def fetch_flight_number(pdf_text: str) -> Optional[str]:
    try:
        # Step 1: Get favourite city from API
        city_resp = requests.get("https://register.hackrx.in/submissions/myFavouriteCity", timeout=10)
        city_resp.raise_for_status()
        fav_city = city_resp.text.strip()
        if not fav_city:
            return None
        print(f"[DEBUG] Favourite city: {fav_city}")

        # Step 2: Find endpoint in PDF for that city
        # Matches the line with the city and the endpoint URL
        pattern = rf"{fav_city}.*?(https?://\S+)"
        match = re.search(pattern, pdf_text, re.IGNORECASE)
        if not match:
            return None
        endpoint_url = match.group(1).strip()
        print(f"[DEBUG] Endpoint found: {endpoint_url}")

        # Step 3: Call endpoint to get flight number
        fn_resp = requests.get(endpoint_url, timeout=10)
        fn_resp.raise_for_status()
        flight_number = fn_resp.text.strip()
        print(f"[DEBUG] Flight number fetched: {flight_number}")

        return flight_number
    except Exception as e:
        print(f"[ERROR] Flight number fetch failed: {e}")
        return None

# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/hackrx/run")
def run_endpoint(req: RunRequest, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {BEARER_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    pdf_path = download_pdf(req.documents)
    pdf_text = extract_text(pdf_path)
    index = build_faiss(pdf_text)

    answers = []
    for q in req.questions:
        ans = run_gpt(q, index)

        if "flight number" in q.lower():
            if not looks_like_flight_number(ans):
                flight_num = fetch_flight_number(pdf_text)
                if flight_num:
                    ans = f"{flight_num}"
        answers.append(ans)

    return {"status": "success", "answers": answers}
