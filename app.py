import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import openai

# CONFIG
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")

INDEX_PATH = "faiss_index"

app = FastAPI(title="HackRx Insurance Q&A API")

# Load FAISS index on startup
embeddings = OpenAIEmbeddings()
try:
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully.")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")
    vectorstore = None

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/hackrx")
async def ask_question(request: QueryRequest):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded. Please run create_index first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve top 5 relevant chunks
    docs = vectorstore.similarity_search(question, k=5)
    if not docs:
        raise HTTPException(status_code=404, detail="No relevant information found in documents.")

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an expert assistant in insurance policy analysis.
Use ONLY the context below to answer the question concisely and accurately.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:
    """

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = completion.choices[0].message["content"]
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {e}")
