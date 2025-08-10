import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import fitz  # PyMuPDF

app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str
    questions: list

# Mapping from PDF
city_to_landmark = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Marina Beach",
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",
    "Pune": "Meenakshi Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    "Paris": "Taj Mahal",
    # Add rest from PDF tables if needed
}

landmark_to_endpoint = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber",
}

@app.post("/hackrx/run")
def run_hackrx(data: HackRxRequest, Authorization: str = Header(None)):
    try:
        # Step 1: Get favourite city
        city = requests.get("https://register.hackrx.in/submissions/myFavouriteCity").text.strip()

        if not city:
            raise HTTPException(status_code=400, detail="Could not fetch favourite city")

        # Step 2: Map city to landmark
        landmark = city_to_landmark.get(city)
        if not landmark:
            raise HTTPException(status_code=400, detail=f"No landmark found for city {city}")

        # Step 3: Determine endpoint
        endpoint = landmark_to_endpoint.get(landmark, "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber")

        # Step 4: Get flight number
        flight_number = requests.get(endpoint).text.strip()

        if not flight_number:
            raise HTTPException(status_code=400, detail="Flight number not found")

        return {"flight_number": flight_number}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
