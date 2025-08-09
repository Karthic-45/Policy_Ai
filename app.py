import os
import tempfile
import asyncio
import requests
import base64
import zipfile
import mimetypes
import pandas as pd
import tarfile
import rarfile
import pytesseract
from PIL import Image
import magic  # pip install python-magic-bin (Windows) or python-magic (Linux/Mac)
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
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.1.0"
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
    print("🔍 Initializing models...")
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
    documents: str  # URL or base64
    questions: List[str]
    is_base64: Optional[bool] = False

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

# Detect file type
def detect_file_type(file_path: str) -> str:
    try:
        mime = magic.from_file(file_path, mime=True)
        guessed_ext = mimetypes.guess_extension(mime) or ""

        with open(file_path, "rb") as f:
            sig = f.read(8)

        if sig.startswith(b"%PDF-"):
            return ".pdf"
        elif sig.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(file_path, 'r') as z:
                names = z.namelist()
                if any(n.startswith("word/") for n in names):
                    return ".docx"
                elif any(n.startswith("xl/") for n in names):
                    return ".xlsx"
                elif any(n.startswith("ppt/") for n in names):
                    return ".pptx"
                else:
                    return ".zip"
        elif sig[:3] == b"\xFF\xD8\xFF":
            return ".jpg"
        elif sig.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        elif sig[:4] == b"Rar!":
            return ".rar"
        elif sig[:5] == b"\x1f\x8b\x08":
            return ".gz"
        elif sig[:4] == b"PK\x05\x06":
            return ".epub"

        return guessed_ext or ".txt"
    except Exception:
        return ".txt"

# OCR helper
def ocr_image(file_path: str) -> str:
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

# Universal loader
def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ["", ".bin"]:
        real_ext = detect_file_type(file_path)
        if real_ext and real_ext != ext:
            new_path = file_path + real_ext
            os.rename(file_path, new_path)
            file_path = new_path
            ext = real_ext

    try:
        if ext == ".pdf":
            docs = PyMuPDFLoader(file_path).load()
            if not any(doc.page_content.strip() for doc in docs):
                # Run OCR on scanned PDF pages (convert to images first if needed)
                return [Document(page_content=ocr_image(file_path))]
            return docs
        elif ext in [".doc", ".docx", ".pptx", ".html", ".htm", ".rtf"]:
            return UnstructuredFileLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path).load()
        elif ext == ".eml":
            return UnstructuredEmailLoader(file_path).load()
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = ocr_image(file_path)
            return [Document(page_content=text)] if text else []
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".zip":
            docs = []
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            docs.extend(load_documents(os.path.join(root, file)))
            return docs
        elif ext == ".rar":
            docs = []
            with tempfile.TemporaryDirectory() as extract_dir:
                with rarfile.RarFile(file_path, 'r') as rar_ref:
                    rar_ref.extractall(extract_dir)
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            docs.extend(load_documents(os.path.join(root, file)))
            return docs
        elif ext in [".tar", ".gz"]:
            docs = []
            with tempfile.TemporaryDirectory() as extract_dir:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    for root, _, files in os.walk(extract_dir):
                        for file in files:
                            docs.extend(load_documents(os.path.join(root, file)))
            return docs
        else:
            with open(file_path, "rb") as f:
                raw_content = f.read()
            try:
                text_content = raw_content.decode("utf-8", errors="ignore")
            except:
                text_content = str(raw_content)
            return [Document(page_content=text_content)]
    except Exception as e:
        raise ValueError(f"Unsupported or unreadable file: {ext} ({e})")

# Async Q&A
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

        if data.is_base64:
            file_bytes = base64.b64decode(data.documents)
        else:
            response = requests.get(data.documents)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download document.")
            file_bytes = response.content

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        docs = load_documents(tmp_path)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]

        if not docs:
            raise HTTPException(status_code=400, detail="No valid content found in document.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        chunks = chunks[:300]

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
