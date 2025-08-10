import os
import re
import tempfile
import requests
import fitz  # PyMuPDF
import logging
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------- CONFIG ----------------
BEARER_TOKEN ="cff339776dc80b453cdfbfa2f4e8dbafe3fa28e3c05fcebba73c46680c8bf594"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FLIGHT_CITY_API = os.getenv("FLIGHT_CITY_API", "https://register.hackrx.in/submissions/myFavouriteCity")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HackRx Flight Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- HELPERS ----------------
def verify_bearer(auth_header: str):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Unauthorized")
    token = auth_header.split("Bearer ")[-1].strip()
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")

def download_pdf(url: str) -> str:
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download PDF.")
    tmp_path = tempfile.mktemp(suffix=".pdf")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
    return tmp_path

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def find_city_and_landmark(pdf_text: str):
    city_match = re.search(r"(?i)(?:city|favourite city)\s*[:\-]\s*([A-Za-z\s]+)", pdf_text)
    landmark_match = re.search(r"(?i)landmark\s*[:\-]\s*([A-Za-z\s]+)", pdf_text)

    city = city_match.group(1).strip() if city_match else None
    landmark = landmark_match.group(1).strip() if landmark_match else None

    return city, landmark

def get_city_from_api():
    try:
        resp = requests.get(FLIGHT_CITY_API, timeout=10)
        data = resp.json()
        return data.get("data", {}).get("city")
    except Exception as e:
        logger.error(f"Error calling city API: {e}")
        return None

def get_flight_number(city: str):
    try:
        flight_api_url = f"https://register.hackrx.in/submissions/flightNumber?city={city}"
        resp = requests.get(flight_api_url, timeout=10)
        data = resp.json()
        return data.get("data", {}).get("flight_number")
    except Exception as e:
        logger.error(f"Error calling flight number API: {e}")
        return None

# ---------------- MAIN ENDPOINT ----------------
@app.post("/hackrx/run")
def hackrx_run(req: HackRxRequest, authorization: str = Header(None)):
    verify_bearer(authorization)
    logger.info(f"Received /hackrx/run for document: {req.documents}")

    pdf_path = download_pdf(req.documents)
    pdf_text = extract_text_from_pdf(pdf_path)

    # Try to find city & landmark in PDF
    city, landmark = find_city_and_landmark(pdf_text)

    if not city:
        logger.info("City not found in PDF, calling city API...")
        city = get_city_from_api()

    if not city:
        raise HTTPException(status_code=400, detail="City or landmark not found in PDF.")

    logger.info(f"Detected city: {city}")

    # Get flight number
    flight_number = get_flight_number(city)
    if not flight_number:
        raise HTTPException(status_code=400, detail="Flight number not found.")

    return {
        "status": "success",
        "answers": [f"Flight number: {flight_number}"]
    }
