import os
import tempfile
import asyncio
import requests
import mimetypes
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

import zipfile
import pandas as pd

# Load env vars
load_dotenv()

app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance questions via RAG + GPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals - set in startup
embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
chat_model_name = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
openai_api_key = os.getenv("OPENAI_API_KEY")
bearer_token = os.getenv("HACKRX_BEARER_TOKEN")

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is missing!")

os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(model=embedding_model_name)
llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)

prompt_template = """
You are an expert assistant in insurance policy analysis.
Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
- Do not make assumptions.
- Quote directly from the policy when possible.

Context:
{context}

Question: {input}
Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)
qa_chain = create_stuff_documents_chain(llm, prompt)

class QuestionRequest(BaseModel):
    question: str

class HackRxRequest(BaseModel):
    documents: str  # URL to document
    questions: List[str]

@app.get("/")
def health_check():
    return {"status": "API is running"}

# Universal loader for many filetypes
def load_documents(file_path: str) -> List:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
        return UnstructuredFileLoader(file_path).load()
    elif ext in [".txt", ".md"]:
        return TextLoader(file_path).load()
    elif ext == ".eml":
        return UnstructuredEmailLoader(file_path).load()
    elif ext in [".png", ".jpg", ".jpeg"]:
        return UnstructuredImageLoader(file_path).load()
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return [Document(page_content=df.to_string())]
    elif ext == ".zip":
        docs = []
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
                for root, _, files in os.walk(extract_dir):
                    for f in files:
                        try:
                            docs.extend(load_documents(os.path.join(root, f)))
                        except Exception as e:
                            print(f"⚠️ Skipped file in ZIP: {f} ({e})")
        return docs
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

async def ask_async(chain, vectorstore, question):
    relevant_chunks = vectorstore.similarity_search(question, k=12)
    raw = await chain.ainvoke({"context": relevant_chunks, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

@app.post("/hackrx/run")
async def hackrx_run(
    data: HackRxRequest, authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != bearer_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        import time
        start_time = time.time()

        # Download document
        resp = requests.get(data.documents)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")

        # Guess extension from content-type or fallback to .bin
        mime_type = resp.headers.get("content-type", "")
        ext = mimetypes.guess_extension(mime_type) or ".bin"

        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Load docs (supports pdf, docx, zip, etc)
        docs = load_documents(tmp_path)
        docs = [d for d in docs if len(d.page_content.strip()) > 200]

        if not docs:
            raise HTTPException(status_code=400, detail="No valid content found in document.")

        # Determine chunk size dynamically based on doc length
        doc_len = len(docs)
        if doc_len <= 5:
            chunk_size = 600
        elif doc_len <= 10:
            chunk_size = 800
        else:
            chunk_size = 1000

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(docs)
        chunks = chunks[:300]  # limit max chunks

        # Create FAISS index
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Run async queries
        tasks = [ask_async(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        print(f"⏱ Total Time: {total_time:.2f} seconds")

        return {"status": "success", "answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")
