import joblib
import pandas as pd
import numpy as np
import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Ensure src is in the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "prediction.log",
    filemode='a',
    format='%(message)s',
    level=logging.INFO
)

# Load model and preprocessing pipeline
model = joblib.load("models/xgboost_model_v1_tuned.joblib")
model_columns = joblib.load("models/model_columns_v1_tuned.joblib")
pipeline = joblib.load("models/fitted_pipeline.joblib")

# Define updated feature lists
numerical_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating'
]

categorical_features = [
    'host_is_superhost', 'instant_bookable', 'room_type', 'neighbourhood_cleansed'
]

binary_amenities = [
    'Has_Wifi', 'Has_Kitchen', 'Has_Heating', 'Has_TV', 'Has_Essentials',
    'Has_Hair_Dryer', 'Has_Iron', 'Has_Free_Parking', 'Has_Hangers', 'Has_Laptop_Friendly_Workspace'
]

all_features = numerical_features + categorical_features + binary_amenities

def predict_price(input_data):
    """
    Predicts Airbnb price using the trained pipeline and model.
    Logs each prediction with a unique request ID.
    """

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Convert input to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise ValueError("Input must be a dict, pd.Series, or pd.DataFrame")

    # Validate required fields
    missing = [f for f in all_features if f not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Ensure correct dtypes
    for field in numerical_features:
        if not np.issubdtype(input_df[field].dtype, np.number):
            raise TypeError(f"Field '{field}' must be numeric")

    for field in categorical_features:
        input_df[field] = input_df[field].astype(str)

    # Ensure binary amenities are 0 or 1
    for field in binary_amenities:
        input_df[field] = input_df[field].fillna(0).astype(int)

    # Clean column names (for XGBoost)
    input_df.columns = input_df.columns.str.replace(r'[\[\]<>]', '_', regex=True)

    # Preprocess
    processed = pipeline.transform(input_df)
    encoded_df = pd.DataFrame(
        processed.toarray() if hasattr(processed, 'toarray') else processed,
        columns=model_columns
)


    # Predict
    prediction = round(float(model.predict(encoded_df)[0]), 2)

    # Log to file
    input_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
    log_msg = (
        f"Request ID: {request_id}\n"
        f"Timestamp : {timestamp}\n"
        f"Input     : {input_str}\n"
        f"Prediction: ${prediction:.2f} CAD\n"
    )
    logging.info(log_msg + "\n")

    # Output
    print(f"\nPredicted price: ${prediction:.2f} CAD (per night)")

    return prediction

# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict Airbnb listing price from JSON input")
    parser.add_argument("input_json", help="Path to JSON file")
    args = parser.parse_args()

    try:
        with open(args.input_json, "r") as f:
            listing_data = json.load(f)
    except Exception as e:
        print(f"Failed to load input JSON: {e}")
        sys.exit(1)

    try:
        predict_price(listing_data)
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(2)
