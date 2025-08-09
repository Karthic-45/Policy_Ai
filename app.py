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

load_dotenv()  # Loads .env if running locally; Railway uses actual env vars

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

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

async def download_file(url: str) -> str:
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {url}")
    
    suffix = os.path.splitext(url)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name

@app.post("/hackrx/run")
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download & load document
    file_path = await download_file(request.documents)
    all_docs = load_documents(file_path)

    # Embeddings & FAISS vector store using your EMBEDDING_MODEL env variable
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM using your CHAT_MODEL env variable (default gpt-3.5-turbo)
    chat_model = os.getenv("CHAT_MODEL", "gpt-4")
    llm = ChatOpenAI(model=chat_model, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the documents:\n\n{context}\n\nQuestion: {question}"
    )
    chain = create_stuff_documents_chain(llm, prompt)

    results = []
    for question in request.questions:
        retrieved_docs = retriever.get_relevant_documents(question)
        answer = chain.run({"context": retrieved_docs, "question": question})
        results.append({"question": question, "answer": answer})

    return {"status": "success", "results": results}
