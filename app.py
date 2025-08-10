import requests
from fastapi import FastAPI, HTTPException, Header, Request

app = FastAPI()

# Full mapping from the provided PDF
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

# Mapping landmark → flight API endpoint
landmark_to_endpoint = {
    "Gateway of India": "https://register.hackrx.in/teams/public/flights/getFirstCityFlightNumber",
    "Taj Mahal": "https://register.hackrx.in/teams/public/flights/getSecondCityFlightNumber",
    "Eiffel Tower": "https://register.hackrx.in/teams/public/flights/getThirdCityFlightNumber",
    "Big Ben": "https://register.hackrx.in/teams/public/flights/getFourthCityFlightNumber"
}

@app.post("/hackrx/run")
async def run_hackrx(request: Request, Authorization: str = Header(None)):
    try:
        # Parse JSON body (flexible - no 422 errors)
        body = await request.json()
        documents = body.get("documents")
        questions = body.get("questions", [])

        # Step 1: Get favourite city from HackRx API
        city_resp = requests.get("https://register.hackrx.in/submissions/myFavouriteCity")
        if city_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Error fetching favourite city")
        city = city_resp.text.strip()

        if not city:
            raise HTTPException(status_code=400, detail="No favourite city found")

        # Step 2: Map city → landmark
        landmark = city_to_landmark.get(city)
        if not landmark:
            raise HTTPException(status_code=400, detail=f"No landmark mapping found for city: {city}")

        # Step 3: Determine flight endpoint
        endpoint = landmark_to_endpoint.get(
            landmark,
            "https://register.hackrx.in/teams/public/flights/getFifthCityFlightNumber"
        )

        # Step 4: Fetch flight number
        flight_resp = requests.get(endpoint)
        if flight_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Error fetching flight number")
        flight_number = flight_resp.text.strip()

        if not flight_number:
            raise HTTPException(status_code=400, detail="Flight number not found")

        # Step 5: Return result
        return {
            "city": city,
            "landmark": landmark,
            "flight_number": flight_number
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
