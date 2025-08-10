import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import logging

app = FastAPI()
logger = logging.getLogger("hackrx")
logging.basicConfig(level=logging.INFO)

class HackRxRequest(BaseModel):
    documents: str
    questions: list

# Cleaned city → landmark mapping (no duplicates)
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
    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Beijing": "Colosseum",
    "Bangkok": "Christ the Redeemer",
    "Toronto": "Burj Khalifa",
    "Dubai": "CN Tower",
    "Amsterdam": "Petronas Towers",
    "Cairo": "Leaning Tower of Pisa",
    "San Francisco": "Mount Fuji",
    "Berlin": "Niagara Falls",
    "Barcelona": "Louvre Museum",
    "Moscow": "Stonehenge",
    "Seoul": "Sagrada Familia",
    "Cape Town": "Acropolis",
    "Istanbul": "Big Ben",
    "Riyadh": "Machu Picchu",
    "Paris": "Taj Mahal",
    "Dubai Airport": "Moai Statues",
    "Singapore": "Christchurch Cathedral",
    "Jakarta": "The Shard",
    "Vienna": "Blue Mosque",
    "Kathmandu": "Neuschwanstein Castle",
    "Los Angeles": "Buckingham Palace",
}

# Landmark → Flight API mapping
landmark_to_endpoint = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber"
}

@app.post("/hackrx/run")
def run_hackrx(data: HackRxRequest, Authorization: str = Header(None)):
    try:
        # Step 1: Fetch favourite city JSON and extract city string
        city_resp = requests.get("https://register.hackrx.in/submissions/myFavouriteCity")
        city_resp.raise_for_status()
        json_data = city_resp.json()
        city = json_data.get("data", {}).get("city")

        if not city:
            raise HTTPException(status_code=400, detail="Could not fetch favourite city")

        logger.info(f"Favourite city received: {city}")

        # Step 2: Map city to landmark
        landmark = city_to_landmark.get(city)
        if not landmark:
            raise HTTPException(status_code=400, detail=f"No landmark found for city {city}")

        logger.info(f"Landmark for city {city}: {landmark}")

        # Step 3: Get flight number endpoint or default
        endpoint = landmark_to_endpoint.get(
            landmark,
            "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
        )

        # Step 4: Fetch flight number
        flight_resp = requests.get(endpoint)
        flight_resp.raise_for_status()
        flight_number = flight_resp.text.strip()

        logger.info(f"Flight number received: {flight_number}")

        if not flight_number:
            raise HTTPException(status_code=400, detail="Flight number not found")

        # Return response
        return {"flight_number": flight_number}

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=502, detail="Error contacting external API")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
