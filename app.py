import os
import tempfile
import asyncio
import requests
import time
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(
    title="HackRx Production API",
    description="Insurance Q&A with GPT + FAISS + OpenAI",
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")

# Models
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the extracted context from the insurance policy to answer the question accurately and concisely.
- Do not make assumptions.
- Quote the policy when possible.

Context:
{context}

Question: {input}
Answer:
""")

qa_chain = create_stuff_documents_chain(llm, prompt)

# Request model
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health check
@app.get("/")
def health():
    return {"status": "up"}

# Async QA function
def batch_embed_chunks(chunks, batch_size=20):
    batched = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    all_embeddings = []
    for batch in batched:
        all_embeddings.extend(embeddings.embed_documents([doc.page_content for doc in batch]))
    return all_embeddings

async def ask_async(llm_chain, vector_store, question):
    rel_chunks = vector_store.similarity_search(question, k=12)
    raw = await llm_chain.ainvoke({"context": rel_chunks, "input": question})

    if isinstance(raw, dict) and "output_text" in raw:
        answer = raw["output_text"]
    elif isinstance(raw, str):
        answer = raw
    else:
        answer = str(raw)

    answer = answer.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# Main route
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split("Bearer ")[1]
    if token != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        start_time = time.time()

        # Download PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Unable to download document.")
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load and clean pages
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]

        # Adaptive chunking
        page_count = len(docs)
        chunk_size = 600 if page_count <= 5 else 800 if page_count <= 10 else 1000
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        chunks = chunks[:300]  # Limit to top 300 chunks

        # FAISS index
        temp_vector_store = FAISS.from_documents(chunks, embeddings)

        # Answer questions
        tasks = [ask_async(qa_chain, temp_vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        duration = round(time.time() - start_time, 2)
        return {
            "status": "success",
            "answers": answers,
           
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
