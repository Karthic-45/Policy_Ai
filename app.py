import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # Load from .env file

# --- 2. FASTAPI APP SETUP ---
app = FastAPI(
    title="Insurance Policy Q&A API",
    description="An API to answer questions about insurance policies using RAG.",
    version="1.0.0"
)

# --- 3. ENABLE CORS FOR ALL ORIGINS (SAFE FOR TESTING) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. LOAD MODELS & FAISS INDEX ON STARTUP ---
vector_store = None
qa_chain = None

try:
    print("🔍 Loading FAISS index and models...")

    # Read keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    chat_model_name = os.getenv("CHAT_MODEL", "gpt-4")

    # Set OpenAI key for LangChain
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Embeddings and vector store
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
    faiss_index_path = "faiss_insurance_index"  # Adjust if needed
    vector_store = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # LLM and QA chain
    llm = ChatOpenAI(model_name=chat_model_name, temperature=0.1)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    print("✅ Index and models loaded successfully.")

except Exception as e:
    print(f"❌ Error loading index or models: {e}")

# --- 5. REQUEST SCHEMA ---
class QuestionRequest(BaseModel):
    question: str

# --- 6. MAIN Q&A ENDPOINT ---
@app.post("/ask", summary="Ask a question about the insurance policies")
def ask_question(request: QuestionRequest):
    if not vector_store or not qa_chain:
        raise HTTPException(status_code=500, detail="Vector store or QA chain not loaded.")
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        print(f"🔎 Processing question: {question}")
        relevant_chunks = vector_store.similarity_search(question, k=4)

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I could not find relevant information in the documents.",
                "source_chunks": []
            }

        response = qa_chain.invoke({
            "input_documents": relevant_chunks,
            "question": question
        })

        return {
            "question": question,
            "answer": response["output_text"],
            "source_chunks": [doc.page_content for doc in relevant_chunks]
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# --- 7. HEALTH CHECK ---
@app.get("/", summary="Root endpoint for health check")
def read_root():
    return {"status": "API is running"}
