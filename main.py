from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
import uvicorn
import uuid
import time
import logging

import mlflow

# Import model serving and metadata from my existing code
from src.predict import (
    predict_price,
    valid_neighbourhoods,
    valid_room_types,
    ensemble_config,
)

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
app = FastAPI(
    title="Airbnb Price Prediction API",
    description="Predict nightly Airbnb prices using a trained ensemble ML model",
    version="1.0.0",
)

# CORS (loose for dev; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g., ["https://my-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# MLflow basic setup (local file store)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("airbnb_price_inference")


# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
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

    # Better example for Swagger
    model_config = {
        "json_schema_extra": {
            "example": {
                "accommodates": 2,
                "bathrooms": 1.0,
                "bedrooms": 1.0,
                "beds": 1.0,
                "minimum_nights": 2,
                "maximum_nights": 1125,
                "number_of_reviews": 35,
                "review_scores_rating": 92.0,
                "host_age_days": 2000,
                "days_since_last_review": 100,
                "avg_review_score": 90.0,
                "season": 1,
                "distance_to_center_km": 2.5,
                "Has_Wifi": 1,
                "Has_Kitchen": 1,
                "Has_Heating": 1,
                "Has_TV": 1,
                "Has_Essentials": 1,
                "Has_Hair_Dryer": 0,
                "Has_Iron": 1,
                "Has_Free_Parking": 0,
                "Has_Hangers": 1,
                "Has_Laptop_Friendly_Workspace": 1,
                "host_has_profile_pic_binary": 1,
                "host_identity_verified_binary": 1,
                "host_is_superhost": 1,
                "instant_bookable": 1,
                "neighbourhood_cleansed": "Downtown",
                "room_type": "Entire home/apt",
            }
        }
    }


# -------------------------------------------------------------------
# Middleware (request tracing and latency logging)
# -------------------------------------------------------------------
@app.middleware("http")
async def add_request_tracing(request, call_next):
    req_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logging.info(
        f"request_id={req_id} path={request.url.path} status={response.status_code} latency_ms={elapsed_ms:.1f}"
    )
    response.headers["X-Request-ID"] = req_id
    return response


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Airbnb Price Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "valid_neighbourhoods": valid_neighbourhoods,
        "valid_room_types": valid_room_types,
    }


@app.post("/predict")
def predict(listing: ListingInput):
    req_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        # Pydantic v2: model_dump() preferred
        input_dict = listing.model_dump()
        price = predict_price(input_dict)

        latency_ms = (time.perf_counter() - started) * 1000.0

        # Log to MLflow
        model_meta = {}
        try:
            model_meta = {
                "weights_rf": ensemble_config["weights"].get("rf"),
                "weights_ridge": ensemble_config["weights"].get("ridge"),
                "weights_xgb": ensemble_config["weights"].get("xgboost"),
            }
        except Exception:
            pass

        with mlflow.start_run(run_name=f"predict_{req_id}"):
            # Keep params primitive for MLflow
            params = {
                k: v for k, v in input_dict.items() if isinstance(v, (int, float, str))
            }
            mlflow.log_params(params)
            if model_meta:
                mlflow.log_params(model_meta)
            mlflow.log_metric("predicted_price_cad", float(price))
            mlflow.log_metric("latency_ms", float(latency_ms))
            mlflow.set_tags(
                {
                    "endpoint": "predict",
                    "request_id": req_id,
                    "room_type": str(input_dict.get("room_type")),
                    "neighbourhood_cleansed": str(
                        input_dict.get("neighbourhood_cleansed")
                    ),
                }
            )

        return {
            "predicted_price": price,
            "currency": "CAD",
            "status": "success",
            "request_id": req_id,
            "latency_ms": round(latency_ms, 2),
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.post("/predict/batch")
def predict_batch(listings: List[ListingInput]):
    req_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        inputs = [li.model_dump() for li in listings]
        predictions = [predict_price(d) for d in inputs]
        latency_ms = (time.perf_counter() - started) * 1000.0

        with mlflow.start_run(run_name=f"predict_batch_{req_id}"):
            mlflow.log_metric("batch_size", len(inputs))
            mlflow.log_metric("latency_ms", float(latency_ms))
            mlflow.set_tags({"endpoint": "predict_batch", "request_id": req_id})

        return {
            "predictions": predictions,
            "currency": "CAD",
            "status": "success",
            "request_id": req_id,
            "latency_ms": round(latency_ms, 2),
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
