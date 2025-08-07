import os
import tempfile
import asyncio
import requests
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Load env
load_dotenv()

# App setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Request schema
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Init models
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1)

prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the following extracted context from the insurance document to answer the question accurately and concisely.
- Quote directly from the policy when possible.
- Do not make assumptions.

Context:
{context}

Question: {input}
Answer:
""")

qa_chain = create_stuff_documents_chain(llm, prompt)

# Async handler
async def ask_async(llm_chain, chunks, embeddings_model, question, top_k=4):
    # Embed all chunk texts
    chunk_texts = [doc.page_content for doc in chunks]
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)

    # Embed question
    question_embedding = embeddings_model.embed_query(question)

    # Cosine similarity
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]

    # Build temporary FAISS
    faiss_store = FAISS.from_documents(top_chunks, embeddings_model)
    relevant_docs = faiss_store.similarity_search_by_vector(question_embedding, k=top_k)

    # Run LLM
    raw = await llm_chain.ainvoke({"context": relevant_docs, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# Endpoint
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

        # Download and save
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load and filter
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 200]

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Precompute chunk embeddings (once)
        chunk_texts = [doc.page_content for doc in chunks]
        chunk_embeddings = embeddings_model.embed_documents(chunk_texts)

        # Embed all questions
        question_embeddings = [embeddings_model.embed_query(q.strip()) for q in data.questions]

        # Score each chunk by max similarity to any question
        sim_matrix = cosine_similarity(question_embeddings, chunk_embeddings)
        max_similarities = np.max(sim_matrix, axis=0)
        top_n = 200
        top_indices = np.argsort(max_similarities)[-top_n:][::-1]
        top_chunks = [chunks[i] for i in top_indices]

        # Parallel async answers
        tasks = [
            ask_async(qa_chain, top_chunks, embeddings_model, q.strip())
            for q in data.questions
        ]
        answers = await asyncio.gather(*tasks)

        print(f"✅ Done in {time.time() - start_time:.2f} seconds")
        return {"status": "success", "answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"status": "running"}
