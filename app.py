import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str
    questions: list

# Mapping from PDF (full)
city_to_landmark = {
    "Delhi": "Gateway of India",
    "Mumbai": "India Gate",
    "Chennai": "Charminar",
    "Hyderabad": "Marina Beach",
    "Ahmedabad": "Howrah Bridge",
    "Mysuru": "Golconda Fort",
    "Kochi": "Qutub Minar",
    "Hyderabad": "Taj Mahal",
    "Pune": "Meenakshi Temple",
    "Nagpur": "Lotus Temple",
    "Chandigarh": "Mysore Palace",
    "Kerala": "Rock Garden",
    "Bhopal": "Victoria Memorial",
    "Varanasi": "Vidhana Soudha",
    "Jaisalmer": "Sun Temple",
    "Pune": "Golden Temple",
    "New York": "Eiffel Tower",
    "London": "Statue of Liberty",
    "Tokyo": "Big Ben",
    "Beijing": "Colosseum",
    "London": "Sydney Opera House",
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
    "Mumbai": "Space Needle",
    "Seoul": "Times Square"
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
        # Step 1: Get favourite city
        city = requests.get("https://register.hackrx.in/submissions/myFavouriteCity").text.strip()
        if not city:
            raise HTTPException(status_code=400, detail="Could not fetch favourite city")

        # Step 2: Map city → landmark
        landmark = city_to_landmark.get(city)
        if not landmark:
            raise HTTPException(status_code=400, detail=f"No landmark found for city {city}")

        # Step 3: Select endpoint based on landmark
        endpoint = landmark_to_endpoint.get(
            landmark,
            "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
        )

        # Step 4: Get flight number
        flight_number = requests.get(endpoint).text.strip()
        if not flight_number:
            raise HTTPException(status_code=400, detail="Flight number not found")

        return {"flight_number": flight_number}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
