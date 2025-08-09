import os
import tempfile
import asyncio
import aiohttp
import base64
import zipfile
import mimetypes
import pandas as pd
import tarfile
import rarfile
import pytesseract
from PIL import Image
import magic
import hashlib
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader, UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Additional ML imports for hybrid approach
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken

# Load environment variables
load_dotenv()

# Enhanced configuration for 1500 pages
MAX_CHUNKS = 2000  # Increased for large documents
CHUNK_SIZE = 600   # Optimized chunk size
CHUNK_OVERLAP = 100
MAX_WORKERS = 16   # Increased parallelism
SIMILARITY_TOP_K = 15  # More context for better accuracy

# Global variables
vector_store_cache = {}
document_cache = {}
tfidf_cache = {}
embeddings = None
llm = None
qa_chain = None
embedding_model = None
tokenizer = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    global embeddings, llm, qa_chain, embedding_model, tokenizer
    
    try:
        print("🚀 Initializing enhanced models...")
        
        # Initialize OpenAI components
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Use faster embedding model for speed
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            chunk_size=1000  # Batch processing
        )
        
        # Use GPT-4 for accuracy
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0,  # More deterministic
            max_tokens=300,
            timeout=15
        )
        
        # Enhanced prompt for insurance documents
        prompt = PromptTemplate.from_template("""
        You are an expert insurance policy analyst with deep knowledge of insurance terminology and regulations.
        
        Based on the provided context from an insurance policy document, answer the question with maximum accuracy.
        
        INSTRUCTIONS:
        1. Provide specific, factual answers with exact numbers, percentages, time periods when available
        2. Quote directly from the policy when possible
        3. If information involves conditions or limitations, include them
        4. If information is not clearly stated in the context, say "Information not specified in the provided policy context"
        5. Be concise but complete
        
        CONTEXT:
        {context}
        
        QUESTION: {input}
        
        ANSWER:
        """)
        
        qa_chain = create_stuff_documents_chain(llm, prompt)
        
        # Initialize sentence transformer for hybrid retrieval
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print("✅ All models initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        raise e
    
    yield
    
    # Cleanup
    executor.shutdown(wait=True)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="HackRx Insurance Q&A API - Enhanced",
    description="High-performance insurance Q&A with 30s response time for 1500+ pages",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class HackRxRequest(BaseModel):
    documents: str  # URL or base64
    questions: List[str]
    is_base64: Optional[bool] = False

class HackRxResponse(BaseModel):
    answers: List[str]

def detect_file_type(file_path: str) -> str:
    """Enhanced file type detection"""
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
                else:
                    return ".zip"
        elif sig[:3] == b"\xFF\xD8\xFF":
            return ".jpg"
        elif sig.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        
        return guessed_ext or ".txt"
    except Exception:
        return ".txt"

def ocr_image(file_path: str) -> str:
    """OCR for image files"""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text.strip()
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

def load_documents_optimized(file_path: str) -> List[Document]:
    """Optimized document loading for large files"""
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
            # Use PyMuPDF for better performance on large PDFs
            docs = PyMuPDFLoader(file_path).load()
            if not any(doc.page_content.strip() for doc in docs):
                # Fallback to OCR if needed
                ocr_text = ocr_image(file_path)
                return [Document(page_content=ocr_text)] if ocr_text else []
            return docs
        elif ext in [".doc", ".docx", ".pptx", ".html", ".htm", ".rtf"]:
            return UnstructuredFileLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding='utf-8').load()
        elif ext in [".png", ".jpg", ".jpeg"]:
            text = ocr_image(file_path)
            return [Document(page_content=text)] if text else []
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            return [Document(page_content=df.to_string())]
        else:
            # Generic fallback
            with open(file_path, "rb") as f:
                raw_content = f.read()
            try:
                text_content = raw_content.decode("utf-8", errors="ignore")
            except:
                text_content = str(raw_content)
            return [Document(page_content=text_content)]
            
    except Exception as e:
        raise ValueError(f"Failed to load file {ext}: {e}")

def create_hybrid_search_index(chunks: List[Document], doc_hash: str):
    """Create both FAISS and TF-IDF indices for hybrid search"""
    try:
        # Create FAISS vector store
        if chunks:
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store_cache[doc_hash] = vector_store
            
            # Create TF-IDF index for keyword matching
            texts = [chunk.page_content for chunk in chunks]
            if texts:
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=2000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
                tfidf_cache[doc_hash] = {
                    'matrix': tfidf_matrix,
                    'vectorizer': tfidf_vectorizer,
                    'chunks': chunks
                }
            
            print(f"✅ Created hybrid index with {len(chunks)} chunks")
            return vector_store
    except Exception as e:
        print(f"⚠️ Index creation failed: {e}")
        return None

def hybrid_retrieval(question: str, doc_hash: str, top_k: int = SIMILARITY_TOP_K) -> List[Document]:
    """Hybrid retrieval combining FAISS and TF-IDF"""
    all_docs = []
    
    try:
        # Strategy 1: FAISS semantic search
        if doc_hash in vector_store_cache:
            vector_store = vector_store_cache[doc_hash]
            semantic_docs = vector_store.similarity_search(question, k=top_k)
            all_docs.extend(semantic_docs)
        
        # Strategy 2: TF-IDF keyword search
        if doc_hash in tfidf_cache:
            cache = tfidf_cache[doc_hash]
            question_tfidf = cache['vectorizer'].transform([question])
            similarities = cosine_similarity(question_tfidf, cache['matrix'])[0]
            
            # Get top scoring chunks
            top_indices = np.argsort(similarities)[-top_k//2:][::-1]
            keyword_docs = [cache['chunks'][i] for i in top_indices if similarities[i] > 0.1]
            all_docs.extend(keyword_docs)
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        return unique_docs[:top_k]
        
    except Exception as e:
        print(f"⚠️ Hybrid retrieval failed: {e}")
        return []

async def download_file_async(url: str) -> bytes:
    """Async file download with timeout"""
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise HTTPException(status_code=400, 
                                     detail=f"Failed to download file: {response.status}")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="File download timeout")

async def process_single_question(question: str, doc_hash: str) -> str:
    """Process individual question with enhanced accuracy"""
    try:
        # Retrieve relevant documents using hybrid search
        relevant_docs = hybrid_retrieval(question, doc_hash, top_k=SIMILARITY_TOP_K)
        
        if not relevant_docs:
            return "Information not specified in the provided policy context."
        
        # Use LangChain QA chain for answer generation
        response = await qa_chain.ainvoke({
            "context": relevant_docs, 
            "input": question
        })
        
        # Clean and validate response
        answer = response.strip()
        if not answer or len(answer) < 10:
            return "Information not specified in the provided policy context."
        
        return answer
        
    except Exception as e:
        print(f"⚠️ Question processing failed: {e}")
        return "Error processing question. Please try again."

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "Enhanced HackRx API is running",
        "version": "2.0.0",
        "cache_size": len(document_cache),
        "vector_stores": len(vector_store_cache)
    }

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    data: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    """Enhanced main endpoint for HackRx - optimized for 1500 pages in 30s"""
    start_time = time.time()
    
    try:
        # Optional: Validate bearer token
        expected_token = os.getenv("HACKRX_BEARER_TOKEN")
        if expected_token:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing Authorization header")
            token = authorization.split("Bearer ")[1]
            if token != expected_token:
                raise HTTPException(status_code=403, detail="Invalid token")
        
        # Generate document hash for caching
        doc_hash = hashlib.md5(data.documents.encode()).hexdigest()
        
        # Process document if not cached
        if doc_hash not in document_cache:
            print(f"📥 Processing new document...")
            
            # Download file
            if data.is_base64:
                file_bytes = base64.b64decode(data.documents)
            else:
                file_bytes = await download_file_async(data.documents)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            try:
                # Load and process documents
                print(f"📄 Loading documents...")
                docs = load_documents_optimized(tmp_path)
                
                # Filter out very short documents
                docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
                
                if not docs:
                    raise HTTPException(status_code=400, detail="No valid content found")
                
                print(f"📄 Found {len(docs)} document sections")
                
                # Split into optimized chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
                )
                
                chunks = splitter.split_documents(docs)
                
                # Limit chunks for performance (keep most relevant ones)
                if len(chunks) > MAX_CHUNKS:
                    # Keep diverse chunks from different parts of document
                    step = len(chunks) // MAX_CHUNKS
                    chunks = chunks[::step][:MAX_CHUNKS]
                
                print(f"📄 Created {len(chunks)} optimized chunks")
                
                # Cache the chunks
                document_cache[doc_hash] = chunks
                
                # Create hybrid search indices
                vector_store = create_hybrid_search_index(chunks, doc_hash)
                
                if not vector_store:
                    raise HTTPException(status_code=500, detail="Failed to create search index")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        processing_time = time.time() - start_time
        print(f"📊 Document processing: {processing_time:.2f}s")
        
        # Process all questions in parallel
        print(f"❓ Processing {len(data.questions)} questions...")
        
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(8)  # Limit concurrent LLM calls
        
        async def controlled_process(question: str) -> str:
            async with semaphore:
                return await process_single_question(question.strip(), doc_hash)
        
        # Execute all questions concurrently
        answers = await asyncio.gather(
            *[controlled_process(q) for q in data.questions],
            return_exceptions=True
        )
        
        # Handle any exceptions
        final_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                print(f"⚠️ Question {i+1} failed: {answer}")
                final_answers.append("Error processing question.")
            else:
                final_answers.append(str(answer))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(data.questions) if data.questions else 0
        
        print(f"✅ Completed {len(final_answers)} answers in {total_time:.2f}s")
        print(f"📊 Average: {avg_time:.2f}s per question")
        
        return HackRxResponse(answers=final_answers)
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Request failed after {total_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker to maintain cache
        loop="asyncio"
    )
