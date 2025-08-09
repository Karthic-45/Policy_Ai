import os
import tempfile
import asyncio
import requests
import zipfile
import mimetypes
import pandas as pd
import logging
from typing import List, Optional
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
    PyMuPDFLoader, UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)

from PIL import Image
import rarfile
import py7zr

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(_name_)
# ------------------------------------------------

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

# Global variables
vector_store = None
qa_chain = None
content_language = None

# Model initialization
try:
    logger.info("🔍 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model_name = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")

    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)

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
    logger.error(f"❌ Error initializing models: {e}")

# Request models
class QuestionRequest(BaseModel):
    question: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health check
@app.get("/")
def health():
    return {"status": "API is running"}

# Question answering route (manual test)
@app.post("/ask")
def ask_question(request: QuestionRequest):
    global content_language
    if not vector_store or not qa_chain:
        logger.error("Model not loaded before /ask request")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    question = request.question.strip()
    if not question:
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    question_lang = detect(question)
    logger.info(f"📥 Received question: {question} | Language: {question_lang}")

    relevant_chunks = vector_store.similarity_search(question, k=12)
    logger.info(f"🔍 Found {len(relevant_chunks)} relevant chunks for the question.")

    response = qa_chain.invoke({
        "context": relevant_chunks,
        "input": question,
        "language": question_lang
    })

    logger.info(f"✅ Answer generated for the question.")
    return {
        "question": question,
        "answer": response
    }

# Document loader
def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    logger.info(f"📂 Loading file: {file_path} | Extension: {ext}")

    try:
        if ext == ".pdf":
            return PyMuPDFLoader(file_path).load()
        elif ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
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
                return [Document(page_content=f"[BINARY XLSX FILE PREVIEW]: {raw[:512].hex()}")]
        elif ext == ".zip":
            return extract_and_load(file_path, zipfile.ZipFile)
        elif ext == ".rar":
            return extract_and_load(file_path, rarfile.RarFile)
        elif ext == ".7z":
            return extract_and_load(file_path, py7zr.SevenZipFile)
        else:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            try:
                decoded = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_data.decode("latin-1", errors="ignore")
            return [Document(page_content=decoded)]
    except Exception as e:
        logger.error(f"❌ Could not read file {file_path}: {e}")
        raise ValueError(f"Could not read file: {file_path} ({e})")

# Helper: extract and load archives
def extract_and_load(file_path, archive_class):
    docs = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with archive_class(file_path) as archive:
            archive.extractall(extract_dir)
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    try:
                        docs.extend(load_documents(full_path))
                    except Exception as e:
                        logger.warning(f"⚠ Skipped file in archive: {file} ({e})")
    return docs

# Async question answering
async def ask_async(llm_chain, vector_store, question):
    global content_language
    lang_code = detect(question)
    logger.info(f"🔎 Processing async question: {question} | Language: {lang_code}")

    # Get top 8 relevant chunks for speed
    rel_chunks = vector_store.similarity_search(question, k=8)
    raw = await llm_chain.ainvoke({
        "context": rel_chunks,
        "input": question,
        "language": lang_code
    })
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        logger.info("⚠ Model could not find a confident answer.")
        return "The policy document does not specify this clearly."
    logger.info("✅ Answer generated successfully.")
    return answer

# Main HackRx API route
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    global content_language
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")

    logger.info("📥 /hackrx/run request received.")

    if not authorization or not authorization.startswith("Bearer "):
        logger.error("❌ Missing or invalid Authorization header.")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        logger.error("❌ Invalid Bearer token.")
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        import time
        start_time = time.time()

        logger.info(f"📄 Downloading document from: {data.documents}")
        response = requests.get(data.documents)
        if response.status_code != 200:
            logger.error(f"❌ Failed to download document. HTTP {response.status_code}")
            raise HTTPException(status_code=400, detail="Failed to download document.")

        mime_type = response.headers.get("content-type", "")
        extension = mimetypes.guess_extension(mime_type) or ".bin"

        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        logger.info(f"✅ Document saved to temporary path: {tmp_path}")

        docs = load_documents(tmp_path)
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]
        logger.info(f"📄 Loaded {len(docs)} documents after filtering.")

        if not docs:
            logger.error("❌ No valid content found in document.")
            raise HTTPException(status_code=400, detail="No valid content found in document.")

        try:
            content_language = detect(docs[0].page_content)
            logger.info(f"🌐 Detected document language: {content_language}")
        except:
            content_language = "unknown"
            logger.warning("⚠ Could not detect document language.")

        # Dynamic chunk size based on doc length
        page_count = len(docs)
        if page_count <= 5:
            chunk_size = 600
        elif page_count <= 50:
            chunk_size = 1000
        else:
            chunk_size = 1500
        logger.info(f"✂ Splitting into chunks with size: {chunk_size}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        chunks = chunks[:500]  # Limit for performance
        logger.info(f"📦 Created {len(chunks)} chunks for vector store.")

        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

        tasks = [ask_async(qa_chain, temp_vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        logger.info(f"✅ Processing complete in {total_time:.2f} seconds.")

        # Return only answers array
        return {"answers": answers}

    except ValueError as ve:
        logger.error(f"❌ {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
