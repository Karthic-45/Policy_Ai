import os
import re
import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

# CONFIG
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment")

os.environ["OPENAI_API_KEY"] = openai_api_key

PDF_FOLDER = "./pdf_documents"
DOCX_FOLDER = "./docx_documents"
TXT_FOLDER = "./txt_documents"
INDEX_PATH = "faiss_index"

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def heading_aware_chunking(text):
    heading_pattern = r"(^[A-Z\s\d\.,\-:]{3,}$)|(^\d+(\.\d+)*\s.+$)"
    lines = text.splitlines()

    chunks = []
    current_heading = "General"
    current_chunk = ""

    for line in lines:
        if re.match(heading_pattern, line.strip()):
            if current_chunk.strip():
                chunks.append(Document(
                    page_content=f"[Heading: {current_heading}]\n{current_chunk.strip()}",
                    metadata={"heading": current_heading}
                ))
                current_chunk = ""
            current_heading = line.strip()
        else:
            current_chunk += " " + line.strip()

    if current_chunk.strip():
        chunks.append(Document(
            page_content=f"[Heading: {current_heading}]\n{current_chunk.strip()}",
            metadata={"heading": current_heading}
        ))

    avg_len = sum(len(d.page_content) for d in chunks) / max(1, len(chunks))
    chunk_size = max(500, min(1500, int(avg_len * 1.2)))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    final_chunks = []
    for doc in chunks:
        splits = splitter.split_text(doc.page_content)
        final_chunks.extend([Document(page_content=part, metadata=doc.metadata) for part in splits])

    return final_chunks

def load_all_documents():
    all_docs = []

    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    for f in pdf_files:
        print(f"Loading PDF: {f}")
        text = extract_text_from_pdf(f)
        all_docs.append(text)

    docx_files = glob.glob(os.path.join(DOCX_FOLDER, "*.docx"))
    for f in docx_files:
        print(f"Loading DOCX: {f}")
        text = extract_text_from_docx(f)
        all_docs.append(text)

    txt_files = glob.glob(os.path.join(TXT_FOLDER, "*.txt"))
    for f in txt_files:
        print(f"Loading TXT: {f}")
        text = extract_text_from_txt(f)
        all_docs.append(text)

    return all_docs

def main():
    texts = load_all_documents()
    if not texts:
        print("No documents found to index.")
        return

    print("Chunking documents with heading awareness...")
    all_chunks = []
    for text in texts:
        chunks = heading_aware_chunking(text)
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    print("Creating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to: {INDEX_PATH}")

if __name__ == "__main__":
    main()
