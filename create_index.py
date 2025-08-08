import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# 2. Get API key and validate
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please check your .env file.")

# Set API key in environment for LangChain to use
os.environ["OPENAI_API_KEY"] = openai_api_key

# 3. Define paths
pdf_folder_path = "./pdf_documents/"         # Folder containing PDFs
faiss_index_path = "faiss_insurance_index"   # Output index path

# 4. Load all PDF documents
def load_all_documents(folder_path):
    all_documents = []
    if not os.path.exists(folder_path):
        print(f"📁 Folder '{folder_path}' does not exist.")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"✅ Loaded: {filename} ({len(docs)} pages)")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
    return all_documents

# 5. Chunk documents into smaller pieces
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

# 6. Create and save FAISS index
def create_and_save_index(chunks, index_path):
    print("🔧 Creating embeddings using OpenAI...")
    embeddings = OpenAIEmbeddings()

    print("📦 Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(index_path)
    print(f"✅ Index saved at: {index_path}")

# 7. Main runner
if __name__ == "__main__":
    start = time.time()

    print("📄 Loading PDF documents...")
    documents = load_all_documents(pdf_folder_path)

    if documents:
        print("✂ Splitting documents into chunks...")
        chunks = chunk_documents(documents)
        print(f"🔢 Total chunks created: {len(chunks)}")

        print("💾 Creating and saving vector index...")
        create_and_save_index(chunks, faiss_index_path)
    else:
        print("⚠ No documents found or failed to load.")

    end = time.time()
    print(f"⏱ Done in {end - start:.2f} seconds")
