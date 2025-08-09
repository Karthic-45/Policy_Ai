import os
import time
import mimetypes
import zipfile
import magic
import tempfile
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ---------------------------
# 1. Load environment variables
# ---------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

os.environ["OPENAI_API_KEY"] = openai_api_key

# ---------------------------
# 2. Paths
# ---------------------------
docs_folder_path = "./documents/"
faiss_index_path = "faiss_universal_index"

# ---------------------------
# 3. File type detection
# ---------------------------
def detect_file_type(file_path):
    mime_type = magic.from_file(file_path, mime=True)
    ext = mimetypes.guess_extension(mime_type) or ""

    with open(file_path, "rb") as f:
        sig = f.read(8)

    if sig.startswith(b"%PDF-"):
        return ".pdf"
    if sig.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(file_path) as z:
            names = z.namelist()
            if any(n.startswith("word/") for n in names):
                return ".docx"
            if any(n.startswith("xl/") for n in names):
                return ".xlsx"
            if any(n.startswith("ppt/") for n in names):
                return ".pptx"
        return ".zip"
    if sig.startswith(b"\xFF\xD8\xFF"):
        return ".jpg"
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    return ext or ".txt"

# ---------------------------
# 4. Universal loader
# ---------------------------
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ["", ".bin"]:
        ext = detect_file_type(file_path)

    try:
        if ext == ".pdf":
            return PyMuPDFLoader(file_path).load()
        elif ext in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(file_path).load()
        elif ext == ".pptx":
            return UnstructuredPowerPointLoader(file_path).load()
        elif ext in [".html", ".htm"]:
            return UnstructuredHTMLLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding="utf-8").load()
        elif ext == ".csv":
            return CSVLoader(file_path).load()
        elif ext == ".xlsx":
            return UnstructuredExcelLoader(file_path).load()
        elif ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return [Document(page_content=text, metadata={"source": file_path})]
        elif ext == ".zip":
            docs = []
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        docs.extend(load_document(os.path.join(root, f)))
            return docs
        else:
            # Fallback: try reading as text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                return [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                print(f"⚠ Cannot read {file_path} as text: {e}")
                return []
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return []

# ---------------------------
# 5. Load all documents from folder
# ---------------------------
def load_all_documents(folder_path):
    all_docs = []
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found.")
        return []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            docs = load_document(file_path)
            if docs:
                all_docs.extend(docs)
                print(f"✅ Loaded: {filename}")
            else:
                print(f"⚠ Skipped: {filename}")
    return all_docs

# ---------------------------
# 6. Chunking
# ---------------------------
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

# ---------------------------
# 7. Create FAISS index
# ---------------------------
def create_and_save_index(chunks, index_path):
    print("🔧 Creating embeddings...")
    embeddings = OpenAIEmbeddings()
    print("📦 Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    print(f"✅ Index saved at: {index_path}")

# ---------------------------
# 8. Main
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()

    print("📄 Loading documents...")
    docs = load_all_documents(docs_folder_path)

    if docs:
        print("✂ Splitting into chunks...")
        chunks = chunk_documents(docs)
        print(f"🔢 Total chunks: {len(chunks)}")

        print("💾 Saving FAISS index...")
        create_and_save_index(chunks, faiss_index_path)
    else:
        print("⚠ No documents found.")

    print(f"⏱ Done in {time.time() - start_time:.2f} seconds")

