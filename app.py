import os
import asyncio
import tempfile
import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

app = FastAPI()

# Prompt Template
prompt_template = PromptTemplate.from_template("""
You are an insurance policy assistant. Use the provided policy context to answer the question accurately.

- Be concise and precise
- Quote directly if relevant
- If the answer is not in the context, say: "The policy document does not specify this clearly."

Context:
{context}

Question: {input}
Answer:
""")

# Chat Model
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0.2)

# Request Schema
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
        import time
        start_time = time.time()

        # Download the PDF and load
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            response = requests.get(data.documents)
            tmp.write(response.content)
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        raw_docs = loader.load()

        # Filter out short/noisy pages
        docs = [doc for doc in raw_docs if len(doc.page_content.strip()) > 150]
        if not docs:
            raise HTTPException(status_code=400, detail="Document has no meaningful content.")

        # Split docs into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        print(f"📄 Total Chunks created: {len(chunks)}")

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created.")

        # Sort by content length and retain most meaningful chunks
        sorted_chunks = sorted(chunks, key=lambda x: len(x.page_content), reverse=True)

        # Divide into manageable batches (e.g., batches of 250)
        batch_size = 250
        chunk_batches = [sorted_chunks[i:i + batch_size] for i in range(0, len(sorted_chunks), batch_size)]

        all_answers = []

        # Process questions on each batch
        for batch_num, batch in enumerate(chunk_batches):
            print(f"Processing batch {batch_num + 1} with {len(batch)} chunks")
            vector_store = FAISS.from_documents(batch, OpenAIEmbeddings(model="text-embedding-ada-002"))

            # Create a fresh QA chain for each batch with retriever
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt_template},
            )

            async def retrieve_answer(q):
                response = await qa_chain.ainvoke({"query": q})
                answer = response.strip()
                if not answer or "i don't know" in answer.lower():
                    return "The policy document does not specify this clearly."
                return answer

            tasks = [retrieve_answer(q.strip()) for q in data.questions]
            batch_answers = await asyncio.gather(*tasks)
            all_answers.append(batch_answers)

        # Merge answers across batches (if needed, deduplicate or keep best)
        merged_answers = all_answers[0]  # assuming same answer repeated, just pick from first batch

        print(f"⏱️ Total Time: {time.time() - start_time:.2f} sec")

        return {
            "status": "success",
            "answers": merged_answers,
            "batches_processed": len(chunk_batches)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
