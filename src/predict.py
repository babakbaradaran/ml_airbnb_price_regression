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

# Import custom preprocessing config
from src.preprocessing import build_preprocessing_pipeline

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
model = joblib.load("models/random_forest_model.joblib")
model_columns = joblib.load("models/model_columns.joblib")
pipeline = joblib.load("models/fitted_pipeline.joblib")

# Feature lists (must match training exactly)
# Define features (must match training)
numerical_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating'
]




categorical_features = [
    'host_is_superhost', 'instant_bookable', 'neighbourhood_cleansed', 'room_type'
]

def predict_price_advanced_module(input_data):
    """
    Predicts Airbnb price using validated input, preprocessing pipeline, and trained model.
    Logs the result with a unique request ID.
    """

    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Convert input to DataFrame
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise ValueError("Input must be a dict, pd.Series, or pd.DataFrame")


    # Validate required fields (excluding 'price')
    required_fields = [f for f in numerical_features + categorical_features if f != 'price']
    missing = [f for f in required_fields if f not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


    # Validate presence of required fields
    required_fields = numerical_features + categorical_features
    missing = [f for f in required_fields if f not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Type-check numerical features
    for field in numerical_features:
        if not np.issubdtype(input_df[field].dtype, np.number):
            raise TypeError(f"Field '{field}' must be numeric")

    # Convert categorical features to string (fix for OneHotEncoder issues)
    for field in categorical_features:
        if field in input_df.columns:
            input_df[field] = input_df[field].astype(str)

    # Preprocess
    processed = pipeline.transform(input_df)
    encoded_df = pd.DataFrame(
        processed.toarray() if hasattr(processed, 'toarray') else processed
    )
    encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = round(float(model.predict(encoded_df)[0]), 2)

    # Console output
    print("\nHow this works:")
    print("- Input is validated and converted to DataFrame.")
    print("- Numerical fields are imputed using means.")
    print("- Categorical fields are one-hot encoded.")
    print("- Result is aligned with model training features.")
    print("- Prediction is made using the trained model.\n")
    print(f"Predicted price for this listing: ${prediction:.2f} CAD")
    print("This is the estimated price per night (in CAD) for the listing described above.")

    # Log to file
    input_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
    log_msg = (
        f"Request ID: {request_id}\n"
        f"Timestamp : {timestamp}\n"
        f"Input     : {input_str}\n"
        f"Prediction: ${prediction:.2f} CAD\n"
    )
    logging.info(log_msg + "\n")

    return prediction

# CLI interface
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict Airbnb listing price from input JSON")
    parser.add_argument("input_json", help="Path to JSON file containing listing data")
    args = parser.parse_args()

    try:
        with open(args.input_json, "r") as f:
            listing_data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON file: {e}")
        sys.exit(1)

    try:
        predicted = predict_price_advanced_module(listing_data)
        print(f"\nPredicted price: ${predicted:.2f} CAD")
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(2)
