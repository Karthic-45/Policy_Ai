import os
import tempfile
import requests
import fitz  # PyMuPDF
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load env vars
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI setup ---
app = FastAPI(title="HackRx RAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    status: str
    answers: List[str]

# --- Helper: Download file ---
def download_file(url: str) -> str:
    logger.info(f"Downloading file from {url}")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download file")
    tmp_path = tempfile.mktemp(suffix=".pdf")
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp_path

# --- Helper: Extract text from PDF page by page ---
def extract_text_from_pdf(pdf_path: str) -> List[Document]:
    logger.info(f"Extracting text from PDF: {pdf_path}")
    docs = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"page": page_num + 1}))
    return docs

# --- Helper: Chunk text for embeddings ---
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- Helper: Create FAISS index ---
def build_faiss_index(docs: List[Document]) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(docs, embeddings)

# --- Helper: Retrieve and answer ---
def retrieve_and_answer(index: FAISS, question: str) -> str:
    retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(question)

    if not relevant_docs:
        return "Flight number not found"

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
You are a strict factual extraction system.
Question: {question}
Context:
{context}

If the answer is explicitly found in the context, return only that exact answer.
If it is not found, respond with: "Flight number not found".
"""
    response = llm.invoke(prompt).content.strip()
    if "not found" in response.lower():
        return "Flight number not found"
    return response

# --- Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
def hackrx_run(req: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        # Step 1: Download PDF
        pdf_path = download_file(req.documents)

        # Step 2: Extract & chunk
        docs = extract_text_from_pdf(pdf_path)
        chunks = chunk_documents(docs)

        # Step 3: Build index
        index = build_faiss_index(chunks)

        # Step 4: Answer each question
        answers = [retrieve_and_answer(index, q) for q in req.questions]

        return HackRxResponse(status="success", answers=answers)

    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))
