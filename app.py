import os
import tempfile
import asyncio
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import (
    PyMuPDFLoader, UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import zipfile
import mimetypes
import pandas as pd
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
vector_store = None
qa_chain = None
try:
    print("\U0001F50D Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model_name = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")

    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)

    prompt = PromptTemplate.from_template("""
    You are an expert assistant in insurance policy analysis.
    Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible. 
    - Do not make assumptions.
    - Quote directly from the policy when possible.

    Context:
    {context}

    Question: {input}
    Answer:
    """)

    qa_chain = create_stuff_documents_chain(llm, prompt)
    print("✅ Models initialized.")
except Exception as e:
    print(f"❌ Error initializing models: {e}")

# Request models
class QuestionRequest(BaseModel):
    question: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health check route
@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not vector_store or not qa_chain:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    relevant_chunks = vector_store.similarity_search(question, k=12)
    response = qa_chain.invoke({"context": relevant_chunks, "input": question})

    return {
        "question": question,
        "answer": response,
        "source_chunks": [doc.page_content for doc in relevant_chunks]
    }

# Universal document loader with fallback
def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    try:
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
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            try:
                                docs.extend(load_documents(full_path))
                            except Exception as e:
                                print(f"⚠ Skipped file in ZIP: {file} ({e})")
            return docs
        else:
            # Fallback for .bin or unknown formats
            with open(file_path, "rb") as f:
                raw_data = f.read()

            try:
                decoded = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_data.decode("latin-1", errors="ignore")

            printable_ratio = sum(c.isprintable() or c.isspace() for c in decoded) / max(1, len(decoded))
            if printable_ratio > 0.7:
                return [Document(page_content=decoded)]
            else:
                hex_preview = raw_data[:512].hex()
                return [Document(page_content=f"[BINARY FILE PREVIEW - first 512 bytes in hex]: {hex_preview}")]
    except Exception as e:
        raise ValueError(f"Could not read file: {file_path} ({e})")

# Async task handler
async def ask_async(llm_chain, vector_store, question):
    rel_chunks = vector_store.similarity_search(question, k=12)
    raw = await llm_chain.ainvoke({"context": rel_chunks, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        import time
        start_time = time.time()

        # Download file
        response = requests.get(data.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")

        # Guess extension
        mime_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(mime_type) or ".bin"

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load documents
        docs = load_documents(tmp_path)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]

        if not docs:
            raise HTTPException(status_code=400, detail="No valid content found in document.")

        # Chunking
        page_count = len(docs)
        chunk_size = 600 if page_count <= 5 else 800 if page_count <= 10 else 1000

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        chunks = chunks[:300]

        # FAISS
        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))

        # Q&A
        tasks = [ask_async(qa_chain, temp_vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        print(f"⏱ Total Time: {total_time:.2f} sec")

        return {
            "status": "success",
            "answers": answers
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
