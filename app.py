import requests
from fastapi import FastAPI, HTTPException, Header, Request
import logging

app = FastAPI()
logger = logging.getLogger("hackrx")
logging.basicConfig(level=logging.INFO)

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

landmark_to_endpoint = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber"
}

@app.post("/hackrx/run")
async def run_hackrx(request: Request, Authorization: str = Header(None)):
    try:
        # Read body without strict validation
        body = await request.json()
        documents = body.get("documents")
        questions = body.get("questions", [])

        # Step 1: Fetch favourite city
        city_resp = requests.get("https://register.hackrx.in/submissions/myFavouriteCity")
        city_resp.raise_for_status()
        json_data = city_resp.json()
        city = json_data.get("data", {}).get("city") or json_data.get("city") or city_resp.text.strip()

        if not city:
            raise HTTPException(status_code=400, detail="Could not fetch favourite city")

        logger.info(f"Favourite city received: {city}")

        # Step 2: Map city → landmark
        landmark = city_to_landmark.get(city)
        if not landmark:
            raise HTTPException(status_code=400, detail=f"No landmark found for city {city}")

        logger.info(f"Landmark for city {city}: {landmark}")

        # Step 3: Choose endpoint
        endpoint = landmark_to_endpoint.get(
            landmark,
            "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
        )

        # Step 4: Fetch flight number
        flight_resp = requests.get(endpoint)
        flight_resp.raise_for_status()

        try:
            # If API returns JSON
            json_flight = flight_resp.json()
            flight_number = json_flight.get("data", {}).get("flightNumber") or json_flight.get("flightNumber")
        except Exception:
            # If plain text
            flight_number = flight_resp.text.strip()

        if not flight_number:
            raise HTTPException(status_code=400, detail="Flight number not found")

        logger.info(f"Flight number received: {flight_number}")

        return {
            "success": True,
            "city": city,
            "landmark": landmark,
            "flight_number": flight_number
        }

    except requests.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise HTTPException(status_code=502, detail="Error contacting external API")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
