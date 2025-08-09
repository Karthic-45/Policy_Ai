import os
import re
import tempfile
import asyncio
import requests
import zipfile
import mimetypes
import pandas as pd
import logging
import fitz  # PyMuPDF
from typing import List, Optional, Iterable
from langdetect import detect

from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import (
    UnstructuredFileLoader, TextLoader,
    UnstructuredEmailLoader, UnstructuredImageLoader
)

from PIL import Image
import rarfile
import py7zr

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ------------------------------------------------

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx Insurance Q&A API",
    description="Answer insurance-related questions using RAG and GPT",
    version="1.0.2"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (kept minimal)
qa_chain = None
content_language = None

# Configurable defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4-1106-preview")
BATCH_SIZE_PAGES = int(os.getenv("BATCH_SIZE_PAGES", "25"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "2500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "50"))

# Model initialization
try:
    logger.info("🔍 Initializing models...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.1)

    prompt = PromptTemplate.from_template("""
You are an expert assistant in insurance policy analysis.
Use the following extracted context from an insurance document to answer the question as accurately and concisely as possible.
- Do not make assumptions.
- Quote directly from the policy when possible.
- Reply in the same language as the question, which is {language}.

Context:
{context}

Question: {input}
Answer:
""")

    qa_chain = create_stuff_documents_chain(llm, prompt)
    logger.info("✅ Models initialized successfully.")
except Exception as e:
    logger.exception("❌ Error initializing models: %s", e)
    raise

# Request models
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# Health check
@app.get("/")
def health():
    return {"status": "API is running"}

# ---------------- Utility loaders (non-PDF) ----------------
def load_non_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".doc", ".docx", ".pptx", ".html", ".htm"]:
            return UnstructuredFileLoader(file_path).load()
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path).load()
        elif ext == ".eml":
            return UnstructuredEmailLoader(file_path).load()
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
            try:
                return UnstructuredImageLoader(file_path).load()
            except Exception:
                with Image.open(file_path) as img:
                    info = f"Image: format={img.format}, size={img.size}, mode={img.mode}"
                return [Document(page_content=info)]
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [Document(page_content=df.to_string())]
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(file_path)
                return [Document(page_content=df.to_string())]
            except Exception:
                with open(file_path, "rb") as f:
                    raw = f.read()
                return [Document(page_content=f"[BINARY XLSX PREVIEW]: {raw[:512].hex()}")]
        else:
            with open(file_path, "rb") as f:
                raw_data = f.read()
            try:
                decoded = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw_data.decode("latin-1", errors="ignore")
            return [Document(page_content=decoded)]
    except Exception as e:
        logger.warning("Non-PDF loader failed for %s: %s", file_path, e)
        return []

# ---------------- PDF streaming (page-by-page) ----------------
def iter_pdf_pages_as_documents(pdf_path: str) -> Iterable[Document]:
    doc = fitz.open(pdf_path)
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text") or ""
            text = text.strip()
            if not text:
                continue
            meta = {"page": pno + 1, "source": os.path.basename(pdf_path)}
            yield Document(page_content=text, metadata=meta)
    finally:
        doc.close()

# ---------------- Archive extractor ----------------
def extract_and_load(file_path, archive_class):
    docs = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with archive_class(file_path) as archive:
            try:
                archive.extractall(extract_dir)
            except Exception:
                archive.extractall(path=extract_dir)
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    ext = os.path.splitext(full_path)[1].lower()
                    if ext == ".pdf":
                        docs.append(Document(page_content=f"[PDF IN ARCHIVE]: {full_path}", metadata={"path": full_path}))
                    else:
                        docs.extend(load_non_pdf(full_path))
    return docs

# ---------------- Incremental FAISS builder ----------------
def build_faiss_index_from_pdf(pdf_path: str,
                               embeddings,
                               chunk_size: int = 1200,
                               chunk_overlap: int = CHUNK_OVERLAP,
                               batch_pages: int = BATCH_SIZE_PAGES,
                               max_chunks: int = MAX_CHUNKS) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    faiss_index = None
    total_chunks = 0
    batch_docs: List[Document] = []
    pages_in_batch = 0

    for page_doc in iter_pdf_pages_as_documents(pdf_path):
        pages_in_batch += 1
        batch_docs.append(page_doc)
        if pages_in_batch >= batch_pages:
            split_chunks = splitter.split_documents(batch_docs)
            split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
            allowed = max_chunks - total_chunks
            if allowed <= 0:
                break
            if len(split_chunks) > allowed:
                split_chunks = split_chunks[:allowed]
            if split_chunks:
                if faiss_index is None:
                    faiss_index = FAISS.from_documents(split_chunks, embeddings)
                else:
                    faiss_index.add_documents(split_chunks)
                total_chunks += len(split_chunks)
            batch_docs = []
            pages_in_batch = 0
            if total_chunks >= max_chunks:
                break
    if pages_in_batch > 0 and total_chunks < max_chunks:
        split_chunks = splitter.split_documents(batch_docs)
        split_chunks = [c for c in split_chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
        allowed = max_chunks - total_chunks
        if len(split_chunks) > allowed:
            split_chunks = split_chunks[:allowed]
        if split_chunks:
            if faiss_index is None:
                faiss_index = FAISS.from_documents(split_chunks, embeddings)
            else:
                faiss_index.add_documents(split_chunks)
            total_chunks += len(split_chunks)
    if faiss_index is None:
        faiss_index = FAISS.from_documents([Document(page_content="")], embeddings)
    return faiss_index

# ---------------- Async QA helper ----------------
async def ask_async_chain(chain, vector_store: FAISS, question: str) -> str:
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    top_k = 6
    docs = vector_store.similarity_search(question, k=top_k)
    if not docs:
        return "The policy document does not specify this clearly."
    raw = await chain.ainvoke({
        "context": docs,
        "input": question,
        "language": lang
    })
    answer = raw.strip()
    if not answer or "i don't know" in answer.lower():
        return "The policy document does not specify this clearly."
    return answer

# ---------------- Puzzle parsing helpers ----------------
# Known landmarks (from the mission brief)
KNOWN_LANDMARKS = [
    "Gateway of India","India Gate","Charminar","Marina Beach","Howrah Bridge","Golconda Fort","Qutub Minar",
    "Taj Mahal","Meenakshi Temple","Lotus Temple","Mysore Palace","Rock Garden","Victoria Memorial","Vidhana Soudha",
    "Sun Temple","Golden Temple","Eiffel Tower","Statue of Liberty","Big Ben","Colosseum","Sydney Opera House",
    "Christ the Redeemer","Burj Khalifa","CN Tower","Petronas Towers","Leaning Tower of Pisa","Mount Fuji",
    "Niagara Falls","Louvre Museum","Stonehenge","Sagrada Familia","Acropolis","Machu Picchu","Moai Statues",
    "Christchurch Cathedral","The Shard","Blue Mosque","Neuschwanstein Castle","Buckingham Palace",
    "Space Needle","Times Square"
]

def extract_city_landmark_mapping_from_pdf(pdf_path: str, max_pages_to_scan: int = 50) -> dict:
    """
    Return a dict mapping city (str) -> landmark (str), extracted from the PDF text.
    Scans up to max_pages_to_scan first pages (or all if smaller).
    """
    mapping = {}
    try:
        with fitz.open(pdf_path) as pdf:
            page_count = pdf.page_count
            scan = min(page_count, max_pages_to_scan)
            raw_lines = []
            for p in range(scan):
                txt = pdf.load_page(p).get_text("text") or ""
                # split lines and store
                for l in txt.splitlines():
                    ln = l.strip()
                    if ln:
                        # remove control characters but keep ASCII content
                        raw_lines.append(ln)
    except Exception as e:
        logger.warning("Could not read PDF for puzzle mapping: %s", e)
        return {}

    # Try to find occurrences of known landmarks in lines and record the city following the landmark
    for line in raw_lines:
        # remove emojis and non-ascii noise for parsing
        ascii_line = re.sub(r'[^\x00-\x7F]+', ' ', line).strip()
        if not ascii_line:
            continue
        for lm in KNOWN_LANDMARKS:
            if lm.lower() in ascii_line.lower():
                # get substring after landmark (first occurrence)
                idx = ascii_line.lower().find(lm.lower())
                after = ascii_line[idx + len(lm):].strip()
                # remove common separators
                after = re.sub(r'^[\-\:\—\–\|,\s]+', '', after)
                # remove trailing punctuation
                after = re.sub(r'[\.,;:\)]*$', '', after).strip()
                # if after is empty, maybe city is before the landmark or on same line in other format; try token approach
                city_candidate = after
                if not city_candidate:
                    # token-based: get tokens and try tokens after the landmark token sequence
                    tokens = ascii_line.split()
                    lm_tokens = lm.split()
                    for i in range(len(tokens) - len(lm_tokens) + 1):
                        if [t.lower() for t in tokens[i:i+len(lm_tokens)]] == [tt.lower() for tt in lm_tokens]:
                            city_tokens = tokens[i+len(lm_tokens):]
                            if city_tokens:
                                city_candidate = " ".join(city_tokens)
                                break
                # final clean
                city_candidate = re.sub(r'[^A-Za-z0-9\s]', '', city_candidate).strip()
                if city_candidate:
                    # store city -> landmark, case-normalize city (strip extra whitespace)
                    mapping[city_candidate] = lm
                break
    # If mapping empty, also attempt pattern matching like "Landmark Current Location" style
    if not mapping:
        # try to find lines that look like "<landmark>    <city>" (two or more spaces delim)
        for line in raw_lines:
            cleaned = re.sub(r'[^\x00-\x7F]+', ' ', line)
            parts = re.split(r'\s{2,}', cleaned)
            if len(parts) >= 2:
                left, right = parts[0].strip(), parts[1].strip()
                for lm in KNOWN_LANDMARKS:
                    if lm.lower() in left.lower():
                        city = re.sub(r'[^A-Za-z0-9\s]', '', right).strip()
                        if city:
                            mapping[city] = lm
    return mapping

def fetch_favorite_city_from_api(timeout: int = 10) -> Optional[str]:
    url = "https://register.hackrx.in/submissions/myFavouriteCity"
    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        logger.warning("Failed to call favouriteCity API: %s", e)
        return None
    if not r.ok:
        logger.warning("FavouriteCity endpoint returned non-OK status: %s", r.status_code)
        return None
    try:
        j = r.json()
    except Exception:
        txt = r.text
        # fallback: try to pull any known city word — caller can handle None
        return None

    # try various keys
    candidate = None
    if isinstance(j, dict):
        # top-level keys
        for k in ("favoriteCity", "favouriteCity", "city", "cityName", "favorite_city"):
            if k in j and j[k]:
                candidate = j[k]
                break
        # nested data
        if not candidate and "data" in j and isinstance(j["data"], dict):
            for k in ("favoriteCity", "favouriteCity", "city", "cityName"):
                if k in j["data"] and j["data"][k]:
                    candidate = j["data"][k]
                    break
    # final normalization
    if isinstance(candidate, str):
        return candidate.strip()
    return None

def choose_flight_endpoint_for_landmark(landmark: str) -> str:
    # endpoints as per brief (full path appended to base)
    endpoint_map = {
        "Gateway of India": "/teams/public/flights/getFirstCityFlightNumber",
        "Taj Mahal": "/teams/public/flights/getSecondCityFlightNumber",
        "Eiffel Tower": "/teams/public/flights/getThirdCityFlightNumber",
        "Big Ben": "/teams/public/flights/getFourthCityFlightNumber",
    }
    return endpoint_map.get(landmark, "/teams/public/flights/getFifthCityFlightNumber")

# ---------------- Main /hackrx/run endpoint ----------------
@app.post("/hackrx/run")
async def hackrx_run(data: HackRxRequest, authorization: Optional[str] = Header(None)):
    global content_language
    expected_token = os.getenv("HACKRX_BEARER_TOKEN")
    logger.info("📥 /hackrx/run request received for document: %s", data.documents)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.split("Bearer ")[1]
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

    tmp_path = None
    try:
        start_time = asyncio.get_event_loop().time()

        # Download (stream) the document like your original flow
        resp = requests.get(data.documents, stream=True, timeout=60)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document.")
        content_type = resp.headers.get("content-type", "")
        extension = mimetypes.guess_extension(content_type.split(";")[0]) or os.path.splitext(data.documents)[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tf:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tf.write(chunk)
            tmp_path = tf.name
        logger.info("✅ Document saved to temporary path: %s", tmp_path)

        # --- PUZZLE DETECTION & SOLVE (robust) ---
        if extension.lower() == ".pdf":
            try:
                # scan first pages (mission brief is small)
                mapping = extract_city_landmark_mapping_from_pdf(tmp_path, max_pages_to_scan=30)
                if mapping:
                    logger.info("Detected city->landmark mapping: %s", mapping)
                    fav_city = fetch_favorite_city_from_api()
                    if fav_city:
                        logger.info("Favourite city from API: %s", fav_city)
                        # try exact then fuzzy match
                        lm = mapping.get(fav_city)
                        if not lm:
                            for city_key, landmark_val in mapping.items():
                                if fav_city.lower() == city_key.lower() or fav_city.lower() in city_key.lower() or city_key.lower() in fav_city.lower():
                                    lm = landmark_val
                                    break
                        if lm:
                            endpoint = choose_flight_endpoint_for_landmark(lm)
                            base = "https://register.hackrx.in"
                            flight_url = base + endpoint
                            logger.info("Calling flight endpoint: %s (landmark=%s)", flight_url, lm)
                            try:
                                r = requests.get(flight_url, timeout=15)
                                if r.ok:
                                    jr = r.json()
                                    # try common shapes
                                    fn = None
                                    if isinstance(jr, dict):
                                        fn = jr.get("data", {}).get("flightNumber") or jr.get("data", {}).get("flightnumber") or jr.get("flightNumber") or jr.get("flightnumber")
                                    if not fn:
                                        # try nested deeper or message parsing fallback
                                        # search for hex-like token in text
                                        txt = r.text
                                        m = re.search(r'([0-9a-fA-F]{4,})', txt)
                                        if m:
                                            fn = m.group(1)
                                    if fn:
                                        return {"answers": [fn]}
                                    else:
                                        # endpoint returned but no flight number found
                                        logger.warning("Flight endpoint returned no flightNumber field. Response: %s", jr)
                                        # fallback: return full response text as single answer (still JSON)
                                        return {"answers": [r.text.strip()]}
                                else:
                                    logger.warning("Flight endpoint returned non-OK status %s", r.status_code)
                            except Exception as e:
                                logger.warning("Calling flight endpoint failed: %s", e)
                        else:
                            logger.info("No landmark matched for favourite city: %s", fav_city)
                    else:
                        logger.info("Could not fetch favourite city from the API.")
            except Exception as e:
                logger.debug("Puzzle detection attempt failed or not a puzzle doc: %s", e)
        # --- END PUZZLE HANDLING ---

        # If not puzzle or puzzle handling didn't return, proceed with original RAG pipeline
        ext = os.path.splitext(tmp_path)[1].lower()
        vector_store = None

        if ext == ".pdf":
            try:
                pdf_doc = fitz.open(tmp_path)
                page_count = pdf_doc.page_count
                pdf_doc.close()
            except Exception:
                page_count = 0

            if page_count == 0:
                chunk_size = 1000
            else:
                if page_count <= 10:
                    chunk_size = 600
                elif page_count <= 200:
                    chunk_size = 1000
                elif page_count <= 800:
                    chunk_size = 1200
                else:
                    chunk_size = 1500

            logger.info("PDF detected (pages=%d). Using chunk_size=%d", page_count, chunk_size)

            vector_store = build_faiss_index_from_pdf(
                pdf_path=tmp_path,
                embeddings=embeddings,
                chunk_size=chunk_size,
                chunk_overlap=CHUNK_OVERLAP,
                batch_pages=BATCH_SIZE_PAGES,
                max_chunks=MAX_CHUNKS
            )
        else:
            # Non-pdf handling
            docs = []
            if ext in [".zip", ".rar", ".7z"]:
                if ext == ".zip":
                    docs = extract_and_load(tmp_path, zipfile.ZipFile)
                elif ext == ".rar":
                    docs = extract_and_load(tmp_path, rarfile.RarFile)
                else:
                    docs = extract_and_load(tmp_path, py7zr.SevenZipFile)
            else:
                docs = load_non_pdf(tmp_path)

            docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= MIN_CHUNK_LEN]
            if not docs:
                raise HTTPException(status_code=400, detail="No readable content found in document.")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            chunks = chunks[:MAX_CHUNKS]
            vector_store = FAISS.from_documents(chunks, embeddings)

        # detect language from first stored chunk if possible
        try:
            first_doc = next(iter(vector_store.docstore._dict.values()), None)
            if first_doc and first_doc.page_content:
                content_language = detect(first_doc.page_content)
            else:
                content_language = "unknown"
        except Exception:
            content_language = "unknown"

        # Answer questions concurrently
        tasks = [ask_async_chain(qa_chain, vector_store, q.strip()) for q in data.questions]
        answers = await asyncio.gather(*tasks)

        total_time = asyncio.get_event_loop().time() - start_time
        logger.info("✅ Processing complete in %.2f seconds. Chunks indexed (approx cap %d).", total_time, MAX_CHUNKS)

        return {"answers": answers}

    except ValueError as ve:
        logger.error("❌ %s", ve)
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Unexpected error: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    finally:
        # cleanup temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
