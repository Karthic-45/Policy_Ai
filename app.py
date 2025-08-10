#!/usr/bin/env python3
import os
import re
import tempfile
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Config
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # replace with actual
PDF_TIMEOUT = 30

app = FastAPI(title="Flight Number Extractor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

def download_pdf(url: str) -> str:
    resp = requests.get(url, stream=True, timeout=PDF_TIMEOUT)
    resp.raise_for_status()
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    for chunk in resp.iter_content(8192):
        tf.write(chunk)
    tf.close()
    return tf.name

def extract_text(pdf_path: str) -> str:
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text())
    doc.close()
    return "\n".join(text)

def find_city_and_landmark(text: str):
    # Example regex — adjust as needed for actual PDF format
    city_match = re.search(r"Your favorite city is\s*:\s*(\w+)", text, re.IGNORECASE)
    landmark_match = re.search(r"Landmark\s*:\s*(.+)", text, re.IGNORECASE)

    city = city_match.group(1).strip() if city_match else None
    landmark = landmark_match.group(1).strip() if landmark_match else None

    return city, landmark

def call_flight_number_api(base_url: str, landmark: str):
    params = {"landmark": landmark}
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    resp = requests.get(base_url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("flight_number") or data.get("flight") or "UNKNOWN"

@app.post("/hackrx/run")
def hackrx_run(data: HackRxRequest, authorization: str = Header(None)):
    # Bearer auth check
    if BEARER_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
        token = authorization.split("Bearer ")[1]
        if token != BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token.")

    try:
        pdf_path = download_pdf(data.documents)
        text = extract_text(pdf_path)

        city, landmark = find_city_and_landmark(text)
        if not city or not landmark:
            raise HTTPException(status_code=400, detail="City or landmark not found in PDF.")

        # Extract API endpoint from PDF text
        url_match = re.search(r"https?://[^\s]+/getFlight", text)
        if not url_match:
            raise HTTPException(status_code=400, detail="Flight API URL not found in PDF.")
        flight_api_url = url_match.group(0)

        flight_number = call_flight_number_api(flight_api_url, landmark)

        return {"flight_number": flight_number}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": "running"}
