import os
import tempfile
import requests
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    JSONLoader, 
    UnstructuredFileLoader
)
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for your given JSON format
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# Loader function
def load_documents(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        return PyMuPDFLoader(file_path).load()

    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()

    elif ext == ".csv":
        return CSVLoader(file_path).load()

    elif ext == ".json":
        return JSONLoader(file_path).load()

    elif ext == ".zip":
        return UnstructuredFileLoader(file_path).load()

    elif ext in [".txt", ".md", ".html", ".pptx", ".xlsx"]:
        return UnstructuredFileLoader(file_path).load()

    elif ext == ".bin":
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            try:
                text = raw_data.decode("utf-8", errors="ignore")
            except UnicodeDecodeError:
                text = raw_data.hex()
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            raise ValueError(f"Failed to load .bin file: {e}")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# File downloader
async def download_file(url: str) -> str:
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {url}")
    
    suffix = os.path.splitext(url)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name

# Main endpoint
@app.post("/hackrx/run")
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {os.getenv('API_KEY')}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download & load the single document
    file_path = await download_file(request.documents)
    all_docs = load_documents(file_path)

    # Create FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the documents:\n\n{context}\n\nQuestion: {question}"
    )
    chain = create_stuff_documents_chain(llm, prompt)

    # Loop through each question
    results = []
    for q in request.questions:
        retrieved_docs = retriever.get_relevant_documents(q)
        answer = chain.run({"context": retrieved_docs, "question": q})
        results.append({"question": q, "answer": answer})

    return {"status": "success", "results": results}
