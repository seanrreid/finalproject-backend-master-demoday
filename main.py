# backend/main.py

from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel
import requests
import logging
from db import Base, engine, SessionLocal, get_db
from models import Location, BusinessIdea, Evaluation
from sqlalchemy.orm import Session
from ml.model import model
from utils.external_api import fetch_trend_data, get_economic_indicator
from utils.financials import estimate_startup_costs, estimate_operational_expenses, generate_p_and_l_statement, estimate_annual_revenue
from utils.risks import identify_potential_risks, assess_risks, suggest_mitigation_strategies
from utils.action_plan import generate_action_plan
import pandas as pd
from typing import List, Optional, Dict, Any
from rapidfuzz import process, fuzz
from utils.config import settings
from sqlalchemy import text  
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import start_http_server

# Import text for raw SQL expressions

def fetch_trend_data(business_idea: str, region: str) -> dict:
    """
    Mock function to fetch trend data for a given business idea and region.
    Replace this with actual implementation.
    """
    # Mock data
    return {"popularity": 75}

def extract_region(location: str) -> str:
    """
    Extracts the region code from a location string.
    Example: 'New York, NY' -> 'US-NY'
    """
    location_mapping = {
        'New York, NY': 'US-NY',
        'Los Angeles, CA': 'US-CA',
        'Chicago, IL': 'US-IL',
        'Houston, TX': 'US-TX',
        'San Francisco, CA': 'US-CA'
        # Add more mappings as needed
    }
    return location_mapping.get(location, 'US')

def get_place_type_and_keyword(business_idea):
    mapping = {
        'accounting firm': {'type': 'accounting', 'keyword': 'accounting'},
        'airport': {'type': 'airport', 'keyword': 'airport'},
        'amusement park': {'type': 'amusement_park', 'keyword': 'amusement park'},
        'aquarium': {'type': 'aquarium', 'keyword': 'aquarium'},
        'art gallery': {'type': 'art_gallery', 'keyword': 'art gallery'},
        'atm': {'type': 'atm', 'keyword': 'atm'},
        'bakery': {'type': 'bakery', 'keyword': 'bakery'},
        'bank': {'type': 'bank', 'keyword': 'bank'},
        'bar': {'type': 'bar', 'keyword': 'bar'},
        'beauty salon': {'type': 'beauty_salon', 'keyword': 'beauty salon'},
        'bicycle store': {'type': 'bicycle_store', 'keyword': 'bicycle store'},
        'book store': {'type': 'book_store', 'keyword': 'book store'},
        'bowling alley': {'type': 'bowling_alley', 'keyword': 'bowling alley'},
        'bus station': {'type': 'bus_station', 'keyword': 'bus station'},
        'cafe': {'type': 'cafe', 'keyword': 'cafe'},
        'car dealer': {'type': 'car_dealer', 'keyword': 'car dealer'},
        'car rental': {'type': 'car_rental', 'keyword': 'car rental'},
        'car repair': {'type': 'car_repair', 'keyword': 'car repair'},
        'car wash': {'type': 'car_wash', 'keyword': 'car wash'},
        'casino': {'type': 'casino', 'keyword': 'casino'},
        'cemetery': {'type': 'cemetery', 'keyword': 'cemetery'},
        'church': {'type': 'church', 'keyword': 'church'},
        'clothing store': {'type': 'clothing_store', 'keyword': 'clothing store'},
        'coffee shop': {'type': 'cafe', 'keyword': 'coffee shop'},
        'comic book store': {'type': 'book_store', 'keyword': 'comic book store'},
        'convenience store': {'type': 'convenience_store', 'keyword': 'convenience store'},
        'courthouse': {'type': 'courthouse', 'keyword': 'courthouse'},
        'dentist': {'type': 'dentist', 'keyword': 'dentist'},
        'department store': {'type': 'department_store', 'keyword': 'department store'},
        'doctor': {'type': 'doctor', 'keyword': 'doctor'},
        'electronics store': {'type': 'electronics_store', 'keyword': 'electronics store'},
        'embassy': {'type': 'embassy', 'keyword': 'embassy'},
        'florist': {'type': 'florist', 'keyword': 'florist'},
        'furniture store': {'type': 'furniture_store', 'keyword': 'furniture store'},
        'gas station': {'type': 'gas_station', 'keyword': 'gas station'},
        'gym': {'type': 'gym', 'keyword': 'gym'},
        'hair salon': {'type': 'hair_care', 'keyword': 'hair salon'},
        'hardware store': {'type': 'hardware_store', 'keyword': 'hardware store'},
        'hospital': {'type': 'hospital', 'keyword': 'hospital'},
        'hotel': {'type': 'lodging', 'keyword': 'hotel'},
        'jewelry store': {'type': 'jewelry_store', 'keyword': 'jewelry store'},
        'laundry': {'type': 'laundry', 'keyword': 'laundry'},
        'lawyer': {'type': 'lawyer', 'keyword': 'lawyer'},
        'library': {'type': 'library', 'keyword': 'library'},
        'liquor store': {'type': 'liquor_store', 'keyword': 'liquor store'},
        'meal delivery': {'type': 'meal_delivery', 'keyword': 'meal delivery'},
        'movie theater': {'type': 'movie_theater', 'keyword': 'movie theater'},
        'museum': {'type': 'museum', 'keyword': 'museum'},
        'night club': {'type': 'night_club', 'keyword': 'night club'},
        'park': {'type': 'park', 'keyword': 'park'},
        'pet store': {'type': 'pet_store', 'keyword': 'pet store'},
        'pharmacy': {'type': 'pharmacy', 'keyword': 'pharmacy'},
        'physiotherapist': {'type': 'physiotherapist', 'keyword': 'physiotherapist'},
        'post office': {'type': 'post_office', 'keyword': 'post office'},
        'real estate agency': {'type': 'real_estate_agency', 'keyword': 'real estate agency'},
        'restaurant': {'type': 'restaurant', 'keyword': 'restaurant'},
        'school': {'type': 'school', 'keyword': 'school'},
        'shoe store': {'type': 'shoe_store', 'keyword': 'shoe store'},
        'shopping mall': {'type': 'shopping_mall', 'keyword': 'shopping mall'},
        'spa': {'type': 'spa', 'keyword': 'spa'},
        'stadium': {'type': 'stadium', 'keyword': 'stadium'},
        'supermarket': {'type': 'supermarket', 'keyword': 'supermarket'},
        'taxi stand': {'type': 'taxi_stand', 'keyword': 'taxi stand'},
        'train station': {'type': 'train_station', 'keyword': 'train station'},
        'travel agency': {'type': 'travel_agency', 'keyword': 'travel agency'},
        'veterinary care': {'type': 'veterinary_care', 'keyword': 'veterinary care'},
        'zoo': {'type': 'zoo', 'keyword': 'zoo'},
        'gymnastics center': {'type': 'gym', 'keyword': 'gymnastics'},
        'dance studio': {'type': 'school', 'keyword': 'dance studio'},
        'yoga studio': {'type': 'gym', 'keyword': 'yoga studio'},
        'art studio': {'type': 'art_gallery', 'keyword': 'art studio'},
        'music school': {'type': 'school', 'keyword': 'music school'},
        'language school': {'type': 'school', 'keyword': 'language school'},
        'auto repair shop': {'type': 'car_repair', 'keyword': 'auto repair'},
        'pet grooming': {'type': 'pet_store', 'keyword': 'pet grooming'},
        'mobile phone store': {'type': 'electronics_store', 'keyword': 'mobile phone store'},
        'optician': {'type': 'health', 'keyword': 'optician'},
        'toy store': {'type': 'store', 'keyword': 'toy store'},
        'gift shop': {'type': 'store', 'keyword': 'gift shop'},
        'butcher shop': {'type': 'store', 'keyword': 'butcher shop'},
        'carpet store': {'type': 'home_goods_store', 'keyword': 'carpet store'},
        'plumber': {'type': 'plumber', 'keyword': 'plumber'},
        'electrician': {'type': 'electrician', 'keyword': 'electrician'},
        'locksmith': {'type': 'locksmith', 'keyword': 'locksmith'},
        'storage facility': {'type': 'storage', 'keyword': 'storage facility'},
        'tattoo shop': {'type': 'store', 'keyword': 'tattoo shop'},
        'chiropractor': {'type': 'health', 'keyword': 'chiropractor'},
        'nail salon': {'type': 'beauty_salon', 'keyword': 'nail salon'},
        'cosmetics store': {'type': 'store', 'keyword': 'cosmetics store'},
        'dry cleaner': {'type': 'laundry', 'keyword': 'dry cleaner'},
        'video game store': {'type': 'store', 'keyword': 'video game store'},
        'brewery': {'type': 'bar', 'keyword': 'brewery'},
        'winery': {'type': 'food', 'keyword': 'winery'},
        'distillery': {'type': 'food', 'keyword': 'distillery'},
        'bed and breakfast': {'type': 'lodging', 'keyword': 'bed and breakfast'},
        'campground': {'type': 'campground', 'keyword': 'campground'},
        'hostel': {'type': 'lodging', 'keyword': 'hostel'},
        'insurance agency': {'type': 'insurance_agency', 'keyword': 'insurance agency'},
        'coworking space': {'type': 'real_estate_agency', 'keyword': 'coworking space'},
        'internet cafe': {'type': 'cafe', 'keyword': 'internet cafe'},
        'printing shop': {'type': 'store', 'keyword': 'printing shop'},
        'candy store': {'type': 'store', 'keyword': 'candy store'},
        'chocolate shop': {'type': 'store', 'keyword': 'chocolate shop'},
        'cookie shop': {'type': 'bakery', 'keyword': 'cookie shop'},
        'ice cream shop': {'type': 'food', 'keyword': 'ice cream shop'},
        'bubble tea shop': {'type': 'cafe', 'keyword': 'bubble tea shop'},
        'juice bar': {'type': 'food', 'keyword': 'juice bar'},
        'hookah lounge': {'type': 'bar', 'keyword': 'hookah lounge'},
        'karaoke bar': {'type': 'bar', 'keyword': 'karaoke bar'},
        'escape room': {'type': 'amusement_park', 'keyword': 'escape room'},
        'arcade': {'type': 'amusement_park', 'keyword': 'arcade'},
        'virtual reality center': {'type': 'amusement_park', 'keyword': 'virtual reality center'},
        'paintball center': {'type': 'amusement_park', 'keyword': 'paintball center'},
        'mini golf course': {'type': 'amusement_park', 'keyword': 'mini golf'},
        'go-kart track': {'type': 'amusement_park', 'keyword': 'go-kart'},
        'tattoo parlor': {'type': 'store', 'keyword': 'tattoo parlor'},
        'piercing studio': {'type': 'store', 'keyword': 'piercing studio'},
        'cannabis dispensary': {'type': 'store', 'keyword': 'cannabis dispensary'},
        'antique store': {'type': 'store', 'keyword': 'antique store'},
        'second-hand store': {'type': 'store', 'keyword': 'second-hand store'},
        'thrift store': {'type': 'store', 'keyword': 'thrift store'},
        'flea market': {'type': 'shopping_mall', 'keyword': 'flea market'},
        'farmers market': {'type': 'shopping_mall', 'keyword': 'farmers market'},
        'butterfly conservatory': {'type': 'zoo', 'keyword': 'butterfly conservatory'},
        'boat rental': {'type': 'travel_agency', 'keyword': 'boat rental'},
        'hiking trail': {'type': 'park', 'keyword': 'hiking trail'},
        'ski resort': {'type': 'lodging', 'keyword': 'ski resort'},
        'surf shop': {'type': 'store', 'keyword': 'surf shop'},
        'scuba diving center': {'type': 'travel_agency', 'keyword': 'scuba diving'},
        'wedding planner': {'type': 'store', 'keyword': 'wedding planner'},
        'event planner': {'type': 'store', 'keyword': 'event planner'},
        'catering service': {'type': 'meal_delivery', 'keyword': 'catering service'},
        'food truck': {'type': 'restaurant', 'keyword': 'food truck'},
        'pet daycare': {'type': 'pet_store', 'keyword': 'pet daycare'},
        'dog walker': {'type': 'pet_store', 'keyword': 'dog walker'},
        'driving school': {'type': 'school', 'keyword': 'driving school'},
        'flight school': {'type': 'school', 'keyword': 'flight school'},
        'private investigator': {'type': 'store', 'keyword': 'private investigator'},
        'security service': {'type': 'store', 'keyword': 'security service'},
        'house cleaning': {'type': 'store', 'keyword': 'house cleaning'},
        'landscaping service': {'type': 'store', 'keyword': 'landscaping service'},
        'gardening service': {'type': 'store', 'keyword': 'gardening service'},
        'photography studio': {'type': 'store', 'keyword': 'photography studio'},
        'graphic design studio': {'type': 'store', 'keyword': 'graphic design studio'},
        'web design agency': {'type': 'store', 'keyword': 'web design agency'},
        'marketing agency': {'type': 'store', 'keyword': 'marketing agency'},
        'consulting firm': {'type': 'store', 'keyword': 'consulting firm'},
        'coworking space': {'type': 'real_estate_agency', 'keyword': 'coworking space'},
        'call center': {'type': 'store', 'keyword': 'call center'},
        'software company': {'type': 'store', 'keyword': 'software company'},
        'game development studio': {'type': 'store', 'keyword': 'game development'},
        'data analysis service': {'type': 'store', 'keyword': 'data analysis'},
        '3d printing service': {'type': 'store', 'keyword': '3d printing'},
        'auto detailing': {'type': 'car_wash', 'keyword': 'auto detailing'},
        'motorcycle repair': {'type': 'car_repair', 'keyword': 'motorcycle repair'},
        'bicycle repair': {'type': 'bicycle_store', 'keyword': 'bicycle repair'},
        'blacksmith': {'type': 'store', 'keyword': 'blacksmith'},
        'pottery studio': {'type': 'art_gallery', 'keyword': 'pottery studio'},
        'ceramics studio': {'type': 'art_gallery', 'keyword': 'ceramics studio'},
        'woodworking shop': {'type': 'store', 'keyword': 'woodworking'},
        'metalworking shop': {'type': 'store', 'keyword': 'metalworking'},
        'tattoo removal': {'type': 'store', 'keyword': 'tattoo removal'},
        'laser hair removal': {'type': 'beauty_salon', 'keyword': 'laser hair removal'},
        'meditation center': {'type': 'health', 'keyword': 'meditation center'},
        'reiki center': {'type': 'health', 'keyword': 'reiki'},
        'nutritionist': {'type': 'health', 'keyword': 'nutritionist'},
        'dietitian': {'type': 'health', 'keyword': 'dietitian'},
        'personal trainer': {'type': 'gym', 'keyword': 'personal trainer'},
        'boxing gym': {'type': 'gym', 'keyword': 'boxing gym'},
        'martial arts school': {'type': 'gym', 'keyword': 'martial arts'},
        'fencing club': {'type': 'gym', 'keyword': 'fencing'},
        'archery range': {'type': 'gym', 'keyword': 'archery'},
        'shooting range': {'type': 'gym', 'keyword': 'shooting range'},
        'horseback riding': {'type': 'travel_agency', 'keyword': 'horseback riding'},
        'animal shelter': {'type': 'pet_store', 'keyword': 'animal shelter'},
        'adoption agency': {'type': 'store', 'keyword': 'adoption agency'},
        'employment agency': {'type': 'store', 'keyword': 'employment agency'},
        'real estate agency': {'type': 'real_estate_agency', 'keyword': 'real estate agency'},
        'mortgage broker': {'type': 'finance', 'keyword': 'mortgage broker'},
        'stockbroker': {'type': 'finance', 'keyword': 'stockbroker'},
        'currency exchange': {'type': 'finance', 'keyword': 'currency exchange'},
        'money transfer service': {'type': 'finance', 'keyword': 'money transfer'},
        'atm': {'type': 'atm', 'keyword': 'atm'},
    }
    return mapping.get(business_idea.lower(), {'type': None, 'keyword': business_idea})


# Prometheus for metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)

# Initialize the database
Base.metadata.create_all(bind=engine)

# Verify database connection
try:
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))  # Explicitly use text for SQL expression
    logging.info("Connected to the SQLite database 'final-project.db' successfully.")
except Exception as e:
    logging.error(f"Failed to connect to the database 'final-project.db': {e}")

# Log configuration status
logging.info(f"GOOGLE_MAPS_API_KEY is set: {bool(settings.google_maps_api_key)}")
logging.info(f"DATABASE_URL is set: {bool(settings.database_url)}")
logging.info(f"API_NINJAS_API_KEY is set: {bool(settings.api_ninjas_key)}")
logging.info(f"EXPLODING_TOPICS_API_KEY is set: {bool(settings.exploding_topics_api_key)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNTER = Counter('api_requests_total', 'Total API Requests')
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API Request Latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_database():
    db = SessionLocal()
    try:
        # Seed locations
        known_locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL"]
        for loc in known_locations:
            if not db.query(Location).filter_by(name=loc).first():
                db.add(Location(name=loc))

        # Seed business ideas
        known_business_ideas = ["Coffee Shop", "Book Store", "Gym"]
        for idea in known_business_ideas:
            if not db.query(BusinessIdea).filter_by(name=idea).first():
                db.add(BusinessIdea(name=idea))

        db.commit()
        logging.info("Database seeding completed.")
    except Exception as e:
        logging.error(f"Error seeding database: {str(e)}", exc_info=True)
        db.rollback()
    finally:
        db.close()

def get_known_locations(db: Session):
    locations = db.query(Location.name).all()
    return [loc[0] for loc in locations]

def get_known_business_ideas(db: Session):
    ideas = db.query(BusinessIdea.name).all()
    return [idea[0] for idea in ideas]

def correct_input(user_input, known_list, threshold=80):
    match, score, _ = process.extractOne(user_input, known_list, scorer=fuzz.WRatio)
    if score >= threshold:
        return match
    else:
        return None

def process_trend_data(trend_data: Dict[str, Any]) -> float:
    if not trend_data:
        return 0.0
    try:
        popularity = trend_data.get("popularity", 0)
        return float(popularity) / 100
    except (ValueError, TypeError) as e:
        logging.error(f"Failed to process trend data: {e}")
        return 0.0

# Define Pydantic models

class FinancialProjection(BaseModel):
    revenue: float
    cost_of_goods_sold: float
    gross_profit: float
    operational_expenses: float
    net_profit: float
    break_even_revenue: float

class Competitor(BaseModel):
    name: str
    rating: float
    user_ratings_total: int
    vicinity: str

class RiskAssessment(BaseModel):
    risk: str
    likelihood: int
    impact: int
    risk_score: int

class EvaluationRequest(BaseModel):
    business_idea: str
    location: str

class EvaluationResponse(BaseModel):
    rating: str
    explanation: str
    competitors: Optional[List[dict]] = []
    corrected_location: str
    corrected_business_idea: str
    new_location_added: bool
    new_business_idea_added: bool
    economic_indicator: float
    financial_projection: Optional[FinancialProjection] = None
    risks: Optional[List[RiskAssessment]] = []
    mitigation_strategies: Optional[List[str]] = []
    business_idea: str
    location: str
    action_plan: Optional[str] = None

class LocationCreate(BaseModel):
    name: str

class BusinessIdeaCreate(BaseModel):
    name: str

@app.on_event("startup")
def startup_event():
    seed_database()
    # Optionally start Prometheus metrics server on a different port
    # start_http_server(8001)  # For example, Prometheus can scrape metrics from port 8001

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, db: Session = Depends(get_db)):
    REQUEST_COUNTER.inc()  # Increment API request counter
    with REQUEST_LATENCY.time():  # Measure latency
        try:
            logging.info(
                f"Received request - Business Idea: {request.business_idea}, Location: {request.location}"
            )
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

        region = extract_region(request.location)
        trend_data = fetch_trend_data(request.business_idea, region=region)
        action_plan = None
        new_location_added = False
        business_idea = request.business_idea
        location = request.location
        new_business_idea_added = False
        known_locations = get_known_locations(db)
        known_business_ideas = get_known_business_ideas(db)

        corrected_location = correct_input(location, known_locations, threshold=80)

        if not corrected_location:
            geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={settings.google_maps_api_key}"
            logging.info(f"Attempting geocoding for new location: {location}")
            try:
                geocode_response = requests.get(geocode_url, timeout=10)
                geocode_response.raise_for_status()
                geocode_data = geocode_response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Geocoding API request failed: {e}")
                raise HTTPException(status_code=503, detail="Geocoding service unavailable.")

            if geocode_data.get('status') != 'OK':
                logging.error(f"Geocoding API error: {geocode_data.get('status')}")
                raise HTTPException(status_code=400, detail='Invalid location provided.')

            # Extract geocode
            lat = geocode_data['results'][0]['geometry']['location']['lat']
            lng = geocode_data['results'][0]['geometry']['location']['lng']
            geo_code = f"{lat},{lng}"

            # Add new location with geo_code
            new_location = Location(name=location, geo_code=geo_code)
            db.add(new_location)
            db.commit()
            corrected_location = location
            new_location_added = True
            logging.info(f"New location added: {location} with geo_code: {geo_code}")
        else:
            new_location_added = False
            logging.info(f"Location correction: {location} -> {corrected_location}")

        # Step 2: Fuzzy match and correct the business idea
        corrected_business_idea = correct_input(
            business_idea, known_business_ideas, threshold=80
        )
        if not corrected_business_idea:
            # Optionally add new business idea
            new_business_idea = BusinessIdea(name=business_idea)
            db.add(new_business_idea)
            db.commit()
            corrected_business_idea = business_idea
            new_business_idea_added = True
            logging.info(f"New business idea added: {business_idea}")
        else:
            new_business_idea_added = False
            logging.info(
                f"Business idea correction: {business_idea} -> {corrected_business_idea}"
            )

        # Step 3: Geocode the corrected location to get latitude and longitude
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={corrected_location}&key={settings.google_maps_api_key}"
        logging.info(f"Attempting geocoding for location: {corrected_location}")
        try:
            geocode_response = requests.get(geocode_url, timeout=10)
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Geocoding API request failed: {e}")
            raise HTTPException(status_code=503, detail="Geocoding service unavailable.")

        if geocode_data.get("status") != "OK":
            logging.error(f"Geocoding API error: {geocode_data.get('status')}")
            raise HTTPException(
                status_code=400, detail="Invalid location provided after correction."
            )

        lat = geocode_data["results"][0]["geometry"]["location"]["lat"]
        lng = geocode_data["results"][0]["geometry"]["location"]["lng"]
        logging.info(f"Successfully geocoded to: {lat}, {lng}")

        # Step 4: Use Places API to find nearby competitors
        # Get the mapped type and keyword
        place_info = get_place_type_and_keyword(corrected_business_idea)

        # Construct the API request parameters
        params = {
            'location': f"{lat},{lng}",
            'radius': 7000,
            'key': settings.google_maps_api_key,
            'keyword': place_info['keyword'],
        }

        # Include the 'type' parameter if it's specified in the mapping
        if place_info['type']:
            params['type'] = place_info['type']

        # Build the API request URL
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

        logging.info(f"Fetching competitors with params: {params}")

        try:
            places_response = requests.get(places_url, params=params, timeout=10)
            places_response.raise_for_status()
            places_data = places_response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Places API request failed: {e}")
            raise HTTPException(status_code=503, detail="Places service unavailable.")

        competitors = []
        if places_data.get("status") == "OK":
            for place in places_data.get("results", []):
                competitors.append({
                    "name": place.get("name"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "vicinity": place.get("vicinity"),
                })
        else:
            logging.warning(f"No competitors found or Places API error: {places_data.get('status')}")

        # Step 5: Fetch external data for enhanced evaluation
        economic_indicator = get_economic_indicator(corrected_location)

        # Step 6: Prepare features for the ML model
        features = {
            "business_idea": corrected_business_idea,
            "location": corrected_location,
            "competitors": len(competitors),
            "economic_indicator": economic_indicator,
        }

        # Convert features to the format expected by the model
        df_features = pd.DataFrame([features])
        df_features_encoded = pd.get_dummies(
            df_features,
            columns=["business_idea", "location"],
            drop_first=True
        )

        # Ensure all model features are present
        model_features = model.feature_names_in_
        for feature in model_features:
            if feature not in df_features_encoded.columns:
                df_features_encoded[feature] = 0

        df_features_encoded = df_features_encoded[model_features]

        # Predict success using the ML model
        prediction = model.predict(df_features_encoded)[0]
        rating = "Great" if prediction == 1 else "Bad"

        # Generate explanation
        explanation = (
            f"The business idea '{corrected_business_idea}' in '{corrected_location}' "
            f"has a predicted success rating of '{rating}'. This is based on current market trends "
            f"and economic indicators."
        )

        # Step 7: Financial Projections
        annual_revenue = estimate_annual_revenue(
            corrected_business_idea,
            corrected_location,
            competitors
        )
        startup_costs = estimate_startup_costs(corrected_business_idea)
        operational_expenses = estimate_operational_expenses(
            corrected_business_idea,
            corrected_location
        )
        cost_of_goods_sold = annual_revenue * 0.5  # Assuming COGS is 30% of revenue
        p_and_l_statement = generate_p_and_l_statement(
            annual_revenue,
            cost_of_goods_sold,
            operational_expenses
        )
        break_even_revenue = startup_costs  # Simplified for this example

        financial_projection = FinancialProjection(
            revenue=p_and_l_statement["revenue"],
            cost_of_goods_sold=p_and_l_statement["cost_of_goods_sold"],
            gross_profit=p_and_l_statement["gross_profit"],
            operational_expenses=p_and_l_statement["operational_expenses"],
            net_profit=p_and_l_statement["net_profit"],
            break_even_revenue=break_even_revenue,
        )

        # Step 8: Risk Assessment
        risks = identify_potential_risks(corrected_business_idea, corrected_location)
        risk_assessment = assess_risks(risks)
        mitigation_strategies = suggest_mitigation_strategies(risk_assessment)

    
    response = EvaluationResponse(
        rating=rating,
        explanation=explanation,
        business_idea=corrected_business_idea,
        location=corrected_location,
        competitors=competitors,
        corrected_location=corrected_location,
        corrected_business_idea=corrected_business_idea,
        new_location_added=new_location_added,
        new_business_idea_added=new_business_idea_added,
        economic_indicator=economic_indicator,
        financial_projection=financial_projection,
        risks=risk_assessment,
        mitigation_strategies=mitigation_strategies,
    )
        
    try:
        action_plan = generate_action_plan(response)
    except Exception as e:
            logging.error(f"Error creating response object: {e}")
            action_plan = "Unable to generate action plan at this time."
           
        
    response.action_plan = action_plan

# Save evaluation to the database
    evaluation = Evaluation(
        business_idea=corrected_business_idea,
        location=corrected_location,
        rating=rating,
        explanation=explanation,
        action_plan=action_plan
        )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    logging.info("Successfully processed the request.")

    return response


@app.post("/locations")
def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    if db.query(Location).filter_by(name=location.name).first():
        raise HTTPException(status_code=400, detail="Location already exists.")
    new_location = Location(name=location.name)
    db.add(new_location)
    db.commit()
    db.refresh(new_location)
    return new_location

@app.post("/business-ideas")
def create_business_idea(business_idea: BusinessIdeaCreate, db: Session = Depends(get_db)):
    if db.query(BusinessIdea).filter_by(name=business_idea.name).first():
        raise HTTPException(status_code=400, detail="Business idea already exists.")
    new_idea = BusinessIdea(name=business_idea.name)
    db.add(new_idea)
    db.commit()
    db.refresh(new_idea)
    return new_idea

@app.get("/routes")
def list_routes():
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append({
                "path": route.path,
                "methods": list(route.methods)
            })
    return routes

@app.get("/locations")
def get_locations(db: Session = Depends(get_db)):
    locations = db.query(Location).all()
    return [loc.name for loc in locations]

@app.get("/business-ideas")
def get_business_ideas(db: Session = Depends(get_db)):
    ideas = db.query(BusinessIdea).all()
    return [idea.name for idea in ideas]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
