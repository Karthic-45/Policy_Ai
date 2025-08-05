import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import tempfile

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- 2. FASTAPI APP SETUP ---
app = FastAPI(
    title="Insurance Policy Q&A API",
    description="An API to answer questions about insurance policies using RAG.",
    version="1.0.0"
)

# --- 3. CORS (Allow All for Testing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. LOAD MODELS & FAISS INDEX ---
vector_store = None
qa_chain = None

try:
    print("\U0001F50D Loading FAISS index and models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model_name = os.getenv("CHAT_MODEL", "gpt-4")

    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    vector_store = FAISS.load_local(
        "faiss_insurance_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)

    prompt = PromptTemplate.from_template("""
    Use the following context to answer the question.

    {context}

    Question: {input}
    """)

    qa_chain = create_stuff_documents_chain(llm, prompt)
    print("✅ Index and models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading index or models: {e}")

# --- 5. Q&A ENDPOINT ---
class QuestionRequest(BaseModel):
    question: str

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

# --- 6. HEALTH CHECK ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

# --- 7. HACKRX EVALUATION ENDPOINT ---
class HackRxRequest(BaseModel):
    documents: str  # Single URL
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(
    data: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        # 1. Download PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        # 2. Load and split
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # 3. Create temp FAISS index
        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))

        answers = []
        for question in data.questions:
            rel_chunks = temp_vector_store.similarity_search(question, k=4)
            answer = qa_chain.invoke({"context": rel_chunks, "input": question})
            answers.append(answer)

        return {
            "status": "success",
            "answers": answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")