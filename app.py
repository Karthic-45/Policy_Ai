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
    PyMuPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import zipfile
import mimetypes
import pandas as pd
import filetype
from langchain.schema import Document
import time

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
    print("🔎 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model_name = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)
    prompt = PromptTemplate.from_template("""
        You are an expert assistant in insurance policy analysis. Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
        - Do not make assumptions.
        - Quote directly from the policy when possible.
        Context: {context}
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

def load_documents(file_path: str) -> List[Document]:
    """
    Loads documents from a given file path, supporting multiple formats.
    """
    file_type = filetype.guess(file_path)
    if file_type:
        ext = file_type.extension
    else:
        # Fallback to extension if filetype cannot guess
        ext = os.path.splitext(file_path)[1].lower().strip(".")

    print(f"🔮 Detected file type: {ext}")

    if ext == "pdf":
        return PyMuPDFLoader(file_path).load()
    elif ext in ["doc", "docx", "pptx"]:
        return UnstructuredFileLoader(file_path).load()
    elif ext in ["txt", "md"]:
        return TextLoader(file_path).load()
    elif ext == "eml":
        return UnstructuredEmailLoader(file_path).load()
    elif ext in ["png", "jpg", "jpeg"]:
        return UnstructuredImageLoader(file_path).load()
    elif ext == "csv":
        df = pd.read_csv(file_path)
        return [Document(page_content=df.to_string())]
    elif ext == "zip":
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
    elif ext in ["bin", "octet-stream", "dat"]:
        print("🤔 Generic binary file detected. Attempting to load as PDF.")
        try:
            return PyMuPDFLoader(file_path).load()
        except Exception as e:
            print(f"❌ Failed to load as PDF: {e}")
            raise ValueError(f"Unsupported file type: {ext}. Could not process.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

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
        start_time = time.time()
        
        # Download file
        response = requests.get(data.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        docs = load_documents(tmp_path)
        
        os.unlink(tmp_path)

        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]
        if not docs:
            raise HTTPException(status_code=400, detail="No valid content found in document.")

        # New dynamic chunking logic
        page_count = len(docs)
        if page_count <= 50:
            chunk_size = 800
            chunk_overlap = 100
        elif page_count <= 200:
            chunk_size = 1200
            chunk_overlap = 150
        else: # For very large documents (>200 pages, including 1500 pages)
            chunk_size = 1500
            chunk_overlap = 200
        
        print(f"📄 Document has {page_count} pages. Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        
        # NOTE: Removed the line `chunks = chunks[:300]` to process all document content.
        
        print(f"🔢 Total chunks created: {len(chunks)}")
        
        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))
        
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
