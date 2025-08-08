import os
import tempfile
import asyncio
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import nltk
from nltk.tokenize import sent_tokenize

# Ensure nltk resources are available
nltk.download('punkt')

# Load environment variables
load_dotenv()

app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM and QA chain
vector_store = None
qa_chain = None
try:
    print("🔍 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"))
    llm = ChatOpenAI(model_name=os.getenv("CHAT_MODEL", "gpt-4-1106-preview"), temperature=0.1)

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


class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]


@app.get("/")
def health():
    return {"status": "API is running"}


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

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 200]

        # Intelligent chunk size based on doc length
        page_count = len(docs)
        chunk_size = 600 if page_count <= 5 else 800 if page_count <= 10 else 1000

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

        # Sentence-preserving splitting
        combined_text = "\n".join([doc.page_content for doc in docs])
        sentences = sent_tokenize(combined_text)
        pseudo_docs = [Document(page_content=s) for s in sentences if len(s.strip()) > 30]
        chunks = text_splitter.split_documents(pseudo_docs)
        print(f"📄 Chunks created: {len(chunks)}")

        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        temp_vector_store = FAISS.from_documents(chunks, embeddings_model)

        # Filter top relevant chunks
        used_chunk_content = set()
        for question in data.questions:
            results = temp_vector_store.similarity_search(question.strip(), k=12)
            for doc in results:
                used_chunk_content.add(doc.page_content)

        filtered_chunks = [doc for doc in chunks if doc.page_content in used_chunk_content]
        if len(filtered_chunks) > 300:
            filtered_chunks = filtered_chunks[:300]

        print(f"📄 Filtered chunks used: {len(filtered_chunks)}")

        optimized_vector_store = FAISS.from_documents(filtered_chunks, embeddings_model)
        tasks = [ask_async(qa_chain, optimized_vector_store, q.strip()) for q in data.questions if q.strip()]
        answers = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        print(f"⏱️ Total Time: {total_time:.2f} sec")

        return {"status": "success", "answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
