import joblib
import pandas as pd
import numpy as np
import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Ensure src is in the path
project_root = Path(__file__).resolve().parent  # Points to ml_airbnb_price_regression/
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Setup logging
log_dir = Path(project_root / "logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "prediction.log",
    filemode='a',
    format='%(message)s',
    level=logging.INFO
)

# Load ensemble model and pipeline from the project root models/ directory
ensemble_config = joblib.load(project_root / "models" / "ensemble_model.joblib")
pipeline = joblib.load(project_root / "models" / "fitted_pipeline.joblib")
model_columns = joblib.load(project_root / "models" / "model_columns.joblib")

# Define feature sets for validation
numerical_features = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds',
    'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating',
    'host_age_days', 'days_since_last_review', 'avg_review_score',
    'season', 'distance_to_center_km'
]

binary_features = [
    'Has_Wifi', 'Has_Kitchen', 'Has_Heating', 'Has_TV', 'Has_Essentials',
    'Has_Hair_Dryer', 'Has_Iron', 'Has_Free_Parking', 'Has_Hangers',
    'Has_Laptop_Friendly_Workspace', 'host_has_profile_pic_binary',
    'host_identity_verified_binary', 'host_is_superhost', 'instant_bookable'
]

def compute_interaction_terms(df):
    """Compute interaction terms for the dataset."""
    df = df.copy()
    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        df['accommodates_bedrooms'] = df['accommodates'] * df['bedrooms']
    if 'minimum_nights' in df.columns and 'number_of_reviews' in df.columns:
        df['min_nights_reviews'] = df['minimum_nights'] * df['number_of_reviews']
    return df

def predict_price(input_data):
    """
    Predicts Airbnb price using the ensemble model (RF, Ridge, XGB).
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

    # Compute interaction terms before validation
    input_df = compute_interaction_terms(input_df)

    # Validate required fields
    required_features = numerical_features + binary_features + ['accommodates_bedrooms', 'min_nights_reviews']
    missing = [f for f in required_features if f not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Ensure correct dtypes
    for field in numerical_features + ['accommodates_bedrooms', 'min_nights_reviews']:
        if field in input_df.columns and not np.issubdtype(input_df[field].dtype, np.number):
            raise TypeError(f"Field '{field}' must be numeric")

    for field in binary_features:
        if field in input_df.columns:
            input_df[field] = input_df[field].fillna(0).astype(int)

    # Add dummy columns for neighbourhood_cleansed and room_type
    for col in model_columns:
        if (col.startswith('neighbourhood_cleansed_') or col.startswith('room_')) and col not in input_df.columns:
            input_df[col] = 0
        if col.startswith('neighbourhood_cleansed_') and 'neighbourhood_cleansed' in input_df.columns:
            if input_df['neighbourhood_cleansed'].iloc[0] == col.split('_', 2)[-1]:
                input_df[col] = 1
        if col.startswith('room_') and 'room_type' in input_df.columns:
            if input_df['room_type'].iloc[0] == col.split('_', 1)[-1]:
                input_df[col] = 1

    # Drop raw categorical columns
    input_df = input_df.drop(columns=['neighbourhood_cleansed', 'room_type'], errors='ignore')

    # Clean column names
    input_df.columns = input_df.columns.str.replace(r'[\[\]<>]', '_', regex=True)

    # Ensure all model_columns are present in the correct order
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Preprocess
    try:
        processed = pipeline.transform(input_df)
        logging.info(f"Request ID {request_id}: Processed shape: {processed.shape}")
    except ValueError as e:
        logging.error(f"Request ID {request_id}: Pipeline transformation failed: {e}")
        raise

    # Predict with each model
    best_rf = ensemble_config['rf_model']
    best_ridge = ensemble_config['ridge_model']
    best_xgb = ensemble_config['xgboost_model']
    weights = ensemble_config['weights']

    y_pred_rf_log = best_rf.predict(processed)
    y_pred_rf = np.expm1(y_pred_rf_log)
    y_pred_ridge_log = best_ridge.predict(processed)
    y_pred_ridge = np.expm1(y_pred_ridge_log)
    y_pred_xgb_log = best_xgb.predict(processed)
    y_pred_xgb_log = np.clip(y_pred_xgb_log, np.log1p(15), np.log1p(24503))
    y_pred_xgb = np.expm1(y_pred_xgb_log)

    # Combine predictions
    prediction = weights['rf'] * y_pred_rf + weights['ridge'] * y_pred_ridge + weights['xgboost'] * y_pred_xgb
    prediction = round(float(prediction[0]), 2)

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