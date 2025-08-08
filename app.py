import os
import time
import asyncio
import logging

import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Configuration for GPU
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx-gpu")

# --- Model and API Configuration ---
# Set device to 'cuda' to leverage the GPU
DEVICE = "cuda"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hkunlp/instructor-large") # Using larger model for better accuracy
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")

# --- Performance & Quality Knobs ---
# With a GPU, we can afford to process the whole document.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_CHUNKS = 5 # Retrieve more chunks for better context for the LLM

# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(
    title="HackRx GPU Version",
    description="Insurance Q&A optimized for GPU execution for maximum accuracy and speed.",
    version="3.0-gpu"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

# -----------------------------
# Core Logic & Helper Functions
# -----------------------------
async def build_retriever_from_url(pdf_url: str):
    """
    Builds a retriever by processing all pages of a PDF from a URL.
    This function is designed to run in a separate thread to not block the FastAPI event loop.
    """
    def process_sync():
        logger.info(f"Downloading PDF from {pdf_url}")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()

        logger.info("Opening PDF and extracting all pages...")
        doc = fitz.open(stream=response.content, filetype="pdf")
        # NOTE: The page triage is removed. We process ALL pages for maximum accuracy.
        docs_to_process = [
            Document(page_content=page.get_text("text"), metadata={"page": i + 1})
            for i, page in enumerate(doc) if page.get_text("text").strip()
        ]
        doc.close()
        logger.info(f"Extracted {len(docs_to_process)} non-empty pages from the document.")

        if not docs_to_process:
            raise ValueError("Document appears to be empty or could not be read.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(docs_to_process)
        logger.info(f"Created {len(chunks)} text chunks.")

        if not chunks:
            raise ValueError("Could not create any text chunks from the document.")

        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' onto GPU device '{DEVICE}'...")
        # This will run on the GPU, making it orders of magnitude faster than the CPU version.
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE}
        )

        logger.info("Creating FAISS vector store on GPU...")
        # FAISS.from_documents will now be significantly faster.
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

    # Run the synchronous, CPU/GPU-bound code in a thread pool
    return await asyncio.to_thread(process_sync)

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "up", "device": DEVICE, "embedding_model": EMBEDDING_MODEL}

@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Header(None)):
    if HACKRX_BEARER_TOKEN and (not authorization or f"Bearer {HACKRX_BEARER_TOKEN}" != authorization):
        raise HTTPException(status_code=403, detail="Invalid token.")

    start_time = time.time()
    try:
        # Pass the PDF URL and questions to the builder
        retriever = await build_retriever_from_url(data.documents)

        llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        async def run_query(question):
            result = await qa_chain.ainvoke({"query": question})
            answer = result.get('result', "No answer found.").strip()
            # A more robust fallback answer
            return answer if answer and "don't know" not in answer.lower() else "The policy document does not seem to contain information about this."

        # Run all questions concurrently
        tasks = [run_query(q) for q in data.questions if q.strip()]
        answers = await asyncio.gather(*tasks)

        duration = round(time.time() - start_time, 2)
        logger.info(f"Request completed in {duration} seconds.")

        return {"status": "success", "answers": answers, "time_sec": duration}

    except Exception as e:
        logger.exception("An error occurred during processing.")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
