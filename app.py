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
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import nltk
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()
nltk.download('punkt')

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

# Pydantic request model
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health check route
@app.get("/")
def health():
    return {"status": "API is running"}

# Async handler for each question
async def ask_async(llm_chain, vector_store, question):
    rel_chunks = vector_store.similarity_search(question, k=12)
    raw = await llm_chain.ainvoke({"context": rel_chunks, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# Sentence-aware chunker
def sentence_chunker(docs, max_chunk_size=1000):
    final_chunks = []

    for doc in docs:
        sentences = sent_tokenize(doc.page_content.strip())
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                current_chunk += " " + sent
            else:
                final_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                current_chunk = sent
        if current_chunk:
            final_chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))

    return final_chunks

# HackRx evaluation endpoint
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

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load and filter document
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 200]

        # Dynamic chunk size based on page count
        page_count = len(docs)
        chunk_size = 600 if page_count <= 5 else 800 if page_count <= 10 else 1000

        # Sentence-aware chunking
        chunks = sentence_chunker(docs, max_chunk_size=chunk_size)
        print(f"📄 Chunks created: {len(chunks)}")

        # Build temporary vector store
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        temp_vector_store = FAISS.from_documents(chunks, embeddings_model)

        # Keep only used chunks
        used_chunk_content = set()
        for question in data.questions:
            results = temp_vector_store.similarity_search(question.strip(), k=12)
            for doc in results:
                used_chunk_content.add(doc.page_content)

        filtered_chunks = [doc for doc in chunks if doc.page_content in used_chunk_content]

        # Cap filtered chunks to 300 for efficiency
        max_chunks = 300
        if len(filtered_chunks) > max_chunks:
            filtered_chunks = filtered_chunks[:max_chunks]

        print(f"📄 Filtered chunks used: {len(filtered_chunks)}")

        # Create final optimized vector store
        optimized_vector_store = FAISS.from_documents(filtered_chunks, embeddings_model)

        # Process all valid questions asynchronously
        tasks = [ask_async(qa_chain, optimized_vector_store, q.strip()) for q in data.questions if q.strip()]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        print(f"⏱️ Total Time: {total_time:.2f} sec")

        return {
            "status": "success",
            "answers": answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
