import os
import asyncio
import pickle
import requests
import tempfile
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import pipeline, logging

# Suppress verbose transformer warnings
logging.set_verbosity_error()

# ======================================================================================
# ===== CONFIG (Tuned for Hackathon Speed) =====
# ======================================================================================
TOP_K_FAISS = 8
TOP_K_BM25 = 8
FINAL_TOP_K = 10
CONTEXT_LIMIT = 3000  # Chars

# Model choices for speed vs. accuracy tradeoff
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Faster reranker
QA_MODEL = "distilbert-base-uncased-distilled-squad"  # Much faster QA model

# ======================================================================================
# ===== FASTAPI APP & IN-MEMORY STORAGE =====
# ======================================================================================
app = FastAPI(
    title="Dynamic Document Q&A API",
    description="Submit a document URL to be indexed, then ask questions.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# In-memory storage for the loaded document indexes
# This simple dictionary will hold the processed data for one document at a time.
document_store: Dict[str, Any] = {}

# ======================================================================================
# ===== LOAD MODELS ON STARTUP =====
# ======================================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load all models once when the server starts to avoid loading them on each request
embeddings_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={'device': device})
reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
qa_pipeline = pipeline("question-answering", model=QA_MODEL, tokenizer=QA_MODEL, device=0 if device == "cuda" else -1)

# Warm-up call to make the first real request faster
print("🔥 Warming up models...")
_ = qa_pipeline(question="warmup", context="warmup")
print("✅ Models are ready.")

# ======================================================================================
# ===== API ENDPOINTS =====
# ======================================================================================
class IndexRequest(BaseModel):
    document_url: str

class QuestionRequest(BaseModel):
    questions: List[str]

@app.get("/", summary="Health Check")
def health_check():
    """Check if the API is running and if a document is indexed."""
    return {
        "status": "API is running",
        "device": device,
        "document_indexed": "document_url" in document_store,
        "indexed_document_url": document_store.get("document_url", "None"),
        "index_chunk_count": document_store.get("chunk_count", 0)
    }

@app.post("/index_document", summary="1. Index a Document")
async def index_document(data: IndexRequest):
    """
    Downloads a PDF from a URL, processes it, and creates searchable indexes in memory.
    This is a slow operation and must be completed before asking questions.
    """
    global document_store
    document_store = {} # Clear previous index

    try:
        print(f"⬇️ Downloading document from: {data.document_url}")
        # Use a temporary file to handle the download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            with requests.get(data.document_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        print("📄 Loading and splitting PDF into chunks...")
        loader = PyMuPDFLoader(tmp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks]

        if not texts:
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        print(f"🧠 Building FAISS and BM25 indexes for {len(texts)} chunks...")
        # Build FAISS vector store
        vector_store = FAISS.from_texts(texts, embeddings_model)
        # Build BM25 sparse retriever
        tokenized_corpus = [doc.lower().split() for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        # Store everything in our global in-memory store
        document_store = {
            "document_url": data.document_url,
            "vector_store": vector_store,
            "bm25": bm25,
            "texts": texts,
            "chunk_count": len(texts)
        }
        print("✅ Indexing complete and ready for questions.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

    return {
        "status": "success",
        "message": f"Document indexed successfully from {data.document_url}",
        "chunks_created": len(texts)
    }

@app.post("/ask", summary="2. Ask Questions")
async def ask_questions(data: QuestionRequest, authorization: Optional[str] = Header(None)):
    """
    Answers questions based on the currently indexed document.
    Requires a document to be indexed first via the /index_document endpoint.
    """
    # --- Authentication (Optional as per your original code) ---
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if expected_token and (not authorization or authorization.split("Bearer ")[-1] != expected_token):
        raise HTTPException(status_code=403, detail="Invalid or missing bearer token.")
    
    # --- Guard Clause: Check if index exists ---
    if "vector_store" not in document_store:
        raise HTTPException(status_code=400, detail="No document has been indexed. Please call /index_document first.")

    # --- Answering Logic ---
    answers = await asyncio.gather(*[answer_question(q) for q in data.questions if q.strip()])
    return {"status": "success", "answers": answers}

# ======================================================================================
# ===== HELPER FUNCTIONS (SEARCH & QA LOGIC) =====
# ======================================================================================
def hybrid_search(query: str) -> List[str]:
    """Performs hybrid search (FAISS + BM25) and reranks the results."""
    # 1. Retrieve from both systems
    faiss_hits = document_store["vector_store"].similarity_search(query, k=TOP_K_FAISS)
    
    bm25_scores = document_store["bm25"].get_scores(query.lower().split())
    bm25_hits = sorted(zip(document_store["texts"], bm25_scores), key=lambda x: x[1], reverse=True)[:TOP_K_BM25]

    # 2. Merge and deduplicate candidates
    candidates = [(doc.page_content, query) for doc in faiss_hits]
    candidates += [(text, query) for text, _ in bm25_hits]
    unique_candidates = list(dict.fromkeys(candidates))

    # 3. Rerank the merged candidates
    scores = reranker.predict(unique_candidates, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(unique_candidates, scores), key=lambda x: x[1], reverse=True)[:FINAL_TOP_K]

    return [text for (text, _), _ in ranked]

async def answer_question(question: str) -> str:
    """Finds context and generates an answer for a single question."""
    context_passages = hybrid_search(question)
    context = " ".join(context_passages)[:CONTEXT_LIMIT]
    
    if not context.strip():
        return "Could not find relevant context to answer the question."
        
    result = qa_pipeline(question=question, context=context)
    return result["answer"].strip()
