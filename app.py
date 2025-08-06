import os
import tempfile
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# --- FastAPI App Setup ---
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Improved API for answering insurance document questions using GPT-4 and RAG.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Persistent FAISS Vector Store (Optional) ---
vector_store = None
qa_chain = None

try:
    print("🔄 Loading static FAISS index and LLM...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.load_local("faiss_insurance_index", embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.1)

    prompt = PromptTemplate.from_template("""
You are an expert insurance policy assistant.

Use the provided policy context to answer the question truthfully and clearly:
- Only use the given context.
- Do not fabricate or assume information.
- Quote the policy when possible.
- If the answer is not clearly mentioned, say: "The policy document does not specify this clearly."

Context:
{context}

Question: {input}
Answer:
""")

    qa_chain = create_stuff_documents_chain(llm, prompt)
    print("✅ Model and chain loaded.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

# --- Health Check ---
@app.get("/")
def root():
    return {"status": "API is running"}

# --- Test Ask Endpoint ---
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not vector_store or not qa_chain:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    relevant_chunks = vector_store.similarity_search(question, k=6)
    raw_answer = qa_chain.invoke({"context": relevant_chunks, "input": question})
    cleaned_answer = raw_answer.strip()

    return {
        "question": question,
        "answer": cleaned_answer,
        "source_chunks": [doc.page_content for doc in relevant_chunks]
    }

# --- HackRx Run Endpoint ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")

    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        print(f"📄 Chunks created: {len(chunks)}")

        temp_vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-ada-002"))

        answers = []
        for q in data.questions:
            q_clean = q.strip()
            rel_chunks = temp_vector_store.similarity_search(q_clean, k=6)
            raw = qa_chain.invoke({"context": rel_chunks, "input": q_clean})
            answer = raw.strip()

            if not answer or "I don't know" in answer.lower():
                answer = "The policy document does not specify this clearly."

            print(f"🔍 Q: {q_clean}\n💬 A: {answer}\n---")
            answers.append(answer)

        return {
            "status": "success",
            "answers": answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
