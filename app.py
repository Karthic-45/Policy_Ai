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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env variables
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

# Load LLM and Embedding Models
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

# Input Models
class QuestionRequest(BaseModel):
    question: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health Check Endpoint
@app.get("/")
def health():
    return {"status": "API is running"}

# Local Test Endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not vector_store or not qa_chain:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    relevant_chunks = vector_store.similarity_search(question, k=4)
    response = qa_chain.invoke({"context": relevant_chunks, "input": question})

    return {
        "question": question,
        "answer": response,
        "source_chunks": [doc.page_content for doc in relevant_chunks]
    }

# Async processing
async def ask_async(llm_chain, vector_store, question):
    rel_chunks = vector_store.similarity_search(question, k=4)
    raw = await llm_chain.ainvoke({"context": rel_chunks, "input": question})
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# HackRx Evaluation Endpoint
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

        # Download the PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load and clean document
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 100]

        if not docs:
            raise HTTPException(status_code=400, detail="The PDF has no valid textual content.")

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        print(f"📄 Chunks created: {len(chunks)}")

        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document.")

        # Limit chunk count
        chunks = chunks[:300]

        # Create temporary vector DB
        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))

        # Answer all questions in parallel
        tasks = [ask_async(qa_chain, temp_vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        print(f"⏱️ Total Time: {time.time() - start_time:.2f} sec")

        return {
            "status": "success",
            "answers": answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
