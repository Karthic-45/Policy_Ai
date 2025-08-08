import os
import hashlib
import tempfile
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import fitz  # PyMuPDF

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# Model and utilities setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings_model = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

prompt_template = PromptTemplate.from_template("""
You are an expert assistant in the insurance domain. Answer the question strictly based on the provided context.

<context>
{context}
</context>

Question: {input}
Answer:
""")

chain = create_stuff_documents_chain(llm, prompt_template)

os.makedirs("./faiss_indexes", exist_ok=True)

def get_sha256(filepath):
    """Compute a unique hash for a given file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def stream_pdf_chunks(file_path, splitter, dedupe_set=None):
    """Yield chunks from PDF one page at a time (with deduplication)."""
    dedupe_set = dedupe_set or set()
    with fitz.open(file_path) as pdf:
        for i in range(pdf.page_count):
            try:
                page_text = pdf.load_page(i).get_text()
            except Exception:
                continue  # skip unreadable pages
            clean_text = page_text.strip()
            if clean_text and (clean_text not in dedupe_set):
                dedupe_set.add(clean_text)
                doc = Document(page_content=clean_text, metadata={"page": i})
                for chunk in splitter.split_documents([doc]):
                    if chunk.page_content.strip():
                        yield chunk

@app.post("/hackrx/run")
async def run_hackrx(data: InputData, authorization: Optional[str] = Header(None)):
    # Auth check
    if not authorization or authorization != f"Bearer {os.getenv('API_KEY')}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download PDF (streamed)
    try:
        response = requests.get(data.documents, timeout=300, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Invalid document URL: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp_path = tmp.name

    try:
        doc_hash = get_sha256(tmp_path)
        faiss_path = f"./faiss_indexes/{doc_hash}"

        if os.path.exists(faiss_path):
            vector = FAISS.load_local(
                faiss_path,
                embeddings_model,
                allow_dangerous_deserialization=True
            )
        else:
            vector = FAISS(embedding_function=embeddings_model)
            BATCH_SIZE = 100
            batch = []
            seen = set()
            for chunk in stream_pdf_chunks(tmp_path, splitter, dedupe_set=seen):
                batch.append(chunk)
                if len(batch) >= BATCH_SIZE:
                    vector.add_documents(batch)
                    batch = []
            if batch:
                vector.add_documents(batch)
            vector.save_local(faiss_path)

        # Multi-question QA
        answers = []
        for question in data.questions:
            docs = vector.similarity_search(question, k=8)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            output = chain.invoke({"input": question, "context": context_text})
            answers.append(output.strip())

        return {"status": "completed", "answers": answers}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
