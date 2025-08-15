from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import uvicorn
import sys
from src.predict import predict_price

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Predict nightly Airbnb prices using a trained ensemble ML model",
    version="1.0.0"
)

class ListingInput(BaseModel):
    accommodates: int = Field(..., ge=1, le=20)
    bathrooms: float = Field(..., gt=0, le=10)
    bedrooms: float = Field(..., ge=0, le=20)
    beds: float = Field(..., ge=0, le=20)
    minimum_nights: int = Field(..., ge=1, le=1000)
    maximum_nights: int = Field(..., ge=1, le=5000)
    number_of_reviews: int = Field(..., ge=0, le=1000)
    review_scores_rating: float = Field(..., ge=0, le=100)
    host_age_days: int = Field(..., ge=0, le=10000)
    days_since_last_review: int = Field(..., ge=0, le=5000)
    avg_review_score: float = Field(..., ge=0, le=100)
    season: Literal[0, 1, 2, 3]
    distance_to_center_km: float = Field(..., ge=0, le=50)

    Has_Wifi: int
    Has_Kitchen: int
    Has_Heating: int
    Has_TV: int
    Has_Essentials: int
    Has_Hair_Dryer: int
    Has_Iron: int
    Has_Free_Parking: int
    Has_Hangers: int
    Has_Laptop_Friendly_Workspace: int
    host_has_profile_pic_binary: int
    host_identity_verified_binary: int
    host_is_superhost: int
    instant_bookable: int

    neighbourhood_cleansed: str
    room_type: Literal["Entire home/apt", "Private room", "Shared room"]

@app.get("/")
def root():
    return {"message": "Airbnb Price Prediction API is running."}

@app.post("/predict")
def predict(listing: ListingInput):
    try:
        input_dict = listing.dict()
        price = predict_price(input_dict)
        return {
            "predicted_price": price,
            "currency": "CAD",
            "status": "success"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Run the server directly for local testing
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
