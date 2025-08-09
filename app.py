import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai

# CONFIG
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")

os.environ["OPENAI_API_KEY"] = openai_api_key

INDEX_PATH = "faiss_index"

app = FastAPI(title="HackRx Insurance Q&A API")

embeddings = OpenAIEmbeddings()
vectorstore = None

try:
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully.")
except Exception as e:
    print(f"❌ Error loading FAISS index: {e}")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/hackrx")
async def ask_question(request: QueryRequest):
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded. Run create_index.py first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

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
