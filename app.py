import os
import tempfile
import asyncio
import requests
import time
import logging
import zipfile
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.retriever import BaseRetriever

# -----------------------------
# Configuration
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(
    title="Unified HackRx Q&A API",
    description="A complete solution for document Q&A with in-memory caching.",
    version="6.0-unified"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Models & API Configuration ---
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hkunlp/instructor-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEVICE = os.getenv("DEVICE", "cuda") # "cuda" for GPU, "cpu" for CPU

# -----------------------------
# Global Objects & Cache
# -----------------------------
try:
    logger.info(f"Initializing models on device: {DEVICE}...")
    # Models are loaded only once when the application starts
    embeddings_model = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": DEVICE})
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

    # In-memory cache to store generated retrievers for each document URL
    retriever_cache: Dict[str, BaseRetriever] = {}
    
    logger.info("System ready.")
except Exception as e:
    logger.critical(f"Failed to initialize the system: {e}", exc_info=True)
    exit()

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str # URL of the document
    questions: List[str]

# -----------------------------
# Core Processing Logic
# -----------------------------
async def create_and_cache_retriever(url: str) -> BaseRetriever:
    """
    Handles the entire 'cold start' process for a new document URL.
    This function downloads, processes, and embeds a document, then caches the resulting retriever.
    """
    
    def process_document_sync():
        # This synchronous function contains all the blocking I/O and CPU/GPU-bound tasks.
        tmp_path = None
        try:
            # 1. Download file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                response = requests.get(url)
                response.raise_for_status()
                tmp.write(response.content)
                tmp_path = tmp.name
            
            # 2. Load document based on file type
            docs = []
            if url.lower().endswith(".pdf"):
                loader = PyMuPDFLoader(tmp_path)
                docs = loader.load()
            elif url.lower().endswith(".docx"):
                loader = Docx2txtLoader(tmp_path)
                docs = loader.load()
            elif url.lower().endswith(".zip"):
                with tempfile.TemporaryDirectory() as extract_dir:
                    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    dir_loader = DirectoryLoader(extract_dir, glob="**/*", use_multithreading=True, loader_map={".pdf": PyMuPDFLoader, ".docx": Docx2txtLoader})
                    docs = dir_loader.load()
            else:
                raise ValueError("Unsupported file type.")

            docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]
            if not docs:
                raise ValueError("Document is empty or contains no processable files.")
            
            # 3. Split documents into chunks
            page_count = len(docs)
            chunk_size = 600 if page_count <= 5 else 800 if page_count <= 10 else 1000
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            
            # 4. Create FAISS vector store (computationally intensive)
            logger.info(f"Creating FAISS index for {url}. This will take several minutes...")
            vector_store = FAISS.from_documents(chunks, embeddings_model)
            
            # 5. Create and return the retriever
            return vector_store.as_retriever(search_kwargs={"k": 5})

        finally:
            # Cleanup the temporary file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Run the blocking function in a separate thread to not freeze the API
    retriever = await asyncio.to_thread(process_document_sync)
    
    # Cache the newly created retriever for future requests
    retriever_cache[url] = retriever
    logger.info(f"Retriever for {url} successfully created and cached.")
    return retriever

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def health():
    return {"status": "up", "device": DEVICE, "cached_documents": list(retriever_cache.keys())}

@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: str = Header(None)):
    if not HACKRX_BEARER_TOKEN or (not authorization or f"Bearer {HACKRX_BEARER_TOKEN}" != authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=403, detail="Invalid or missing token.")
    
    doc_url = data.documents
    start_time = time.time()
    
    try:
        # Check if the retriever for this URL is already in the cache
        if doc_url in retriever_cache:
            # Fast Path: Use the cached retriever
            logger.info(f"Cache HIT for URL: {doc_url}")
            retriever = retriever_cache[doc_url]
        else:
            # Slow Path: Process and cache the new document
            logger.info(f"Cache MISS for URL: {doc_url}. Starting processing...")
            retriever = await create_and_cache_retriever(doc_url)
        
        # Create the QA chain using the obtained retriever
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Process all questions
        answers = []
        for question in data.questions:
            if not question.strip():
                continue
            
            result = qa_chain.invoke({"query": question})
            answer = result.get('result', "No answer found.").strip()
            
            if "don't know" in answer.lower():
                answers.append("The policy document does not seem to contain information about this.")
            else:
                answers.append(answer)

        duration = round(time.time() - start_time, 2)
        logger.info(f"Request for {doc_url} completed in {duration} seconds.")

        return {"status": "success", "answers": answers}

    except Exception as e:
        logger.error(f"An error occurred while processing request for {doc_url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
