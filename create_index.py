# create_index.py
import os
import tempfile
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def extract_and_chunk(pdf_path, chunk_size=1200, chunk_overlap=100):
    """Load PDF, chunk text structure-aware, return chunks."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Structure-aware chunking: respects headings & paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        length_function=len
    )
    return splitter.split_documents(docs)


def create_faiss_index(pdf_paths):
    """Create FAISS index for multiple PDFs efficiently."""
    all_chunks = []

    # Process PDFs in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_and_chunk, pdf_paths)
        for chunks in results:
            all_chunks.extend(chunks)

    # Create embeddings + FAISS index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(all_chunks, embeddings)
    return vector_store


def build_index_from_urls(pdf_urls):
    """Download multiple PDFs, build FAISS index in temp dir."""
    temp_files = []

    try:
        # Download PDFs to temp dir
        for url in pdf_urls:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(requests.get(url).content)
            temp_file.flush()
            temp_files.append(temp_file.name)

        # Adaptive chunk size for big PDFs
        pdf_chunk_size = []
        for f in temp_files:
            doc = fitz.open(f)
            pages = len(doc)
            if pages > 1000:
                pdf_chunk_size.append(1800)
            elif pages > 500:
                pdf_chunk_size.append(1500)
            else:
                pdf_chunk_size.append(1200)

        # Create vector store with adaptive sizes
        all_chunks = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(extract_and_chunk, temp_files[i], pdf_chunk_size[i])
                for i in range(len(temp_files))
            ]
            for f in futures:
                all_chunks.extend(f.result())

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        return vector_store

    finally:
        # Cleanup
        for f in temp_files:
            try:
                os.unlink(f)
            except:
                pass
