import joblib
import pandas as pd
import numpy as np
import logging
import sys
import uuid
from pathlib import Path
from datetime import datetime
from src.preprocessing import compute_interaction_terms

# Resolve project root (ml_airbnb_price_regression/)
project_root = Path().resolve()
while not (project_root / "models").exists():
    project_root = project_root.parent
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find 'models' directory in project structure")
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "prediction.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load ensemble model and pipeline
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
dummy_features = [col for col in model_columns if col.startswith('neighbourhood_cleansed_') or col.startswith('room_')]
valid_neighbourhoods = [col.split('_', 2)[-1] for col in model_columns if col.startswith('neighbourhood_cleansed_')]
valid_room_types = ['Entire home/apt', 'Private room', 'Shared room']

def validate_input(df):
    """Validate input data ranges and categorical values."""
    for field in numerical_features:
        if field in df.columns:
            if field in ['accommodates', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'host_age_days', 'days_since_last_review']:
                if (df[field] < 0).any():
                    raise ValueError(f"Field '{field}' cannot be negative")
            if field == 'bathrooms' and (df[field] <= 0).any():
                raise ValueError(f"Field 'bathrooms' must be positive")
            if field in ['review_scores_rating', 'avg_review_score'] and ((df[field] < 0).any() or (df[field] > 100).any()):
                raise ValueError(f"Field '{field}' must be between 0 and 100")
            if field == 'season' and not df[field].isin([0, 1, 2, 3]).all():
                raise ValueError(f"Field 'season' must be in [0, 1, 2, 3]")
            if field == 'distance_to_center_km' and (df[field] < 0).any():
                raise ValueError(f"Field 'distance_to_center_km' cannot be negative")

    if 'neighbourhood_cleansed' in df.columns:
        invalid_neighbourhoods = df['neighbourhood_cleansed'][~df['neighbourhood_cleansed'].isin(valid_neighbourhoods)]
        if not invalid_neighbourhoods.empty:
            raise ValueError(f"Invalid neighbourhood(s): {invalid_neighbourhoods.tolist()}")

    if 'room_type' in df.columns:
        invalid_room_types = df['room_type'][~df['room_type'].isin(valid_room_types)]
        if not invalid_room_types.empty:
            raise ValueError(f"Invalid room type(s): {invalid_room_types.tolist()}")

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
        logging.error(f"Request ID {request_id}: Invalid input type: {type(input_data)}")
        raise ValueError("Input must be a dict, pd.Series, or pd.DataFrame")

    # Compute interaction terms
    try:
        input_df = compute_interaction_terms(input_df)
        logging.info(f"Request ID {request_id}: Interaction terms computed")
    except Exception as e:
        logging.error(f"Request ID {request_id}: Interaction terms computation failed: {e}")
        raise

    # Validate input
    required_features = numerical_features + binary_features + ['accommodates_bedrooms', 'min_nights_reviews']
    missing = [f for f in required_features if f not in input_df.columns]
    if missing:
        logging.error(f"Request ID {request_id}: Missing required fields: {missing}")
        raise ValueError(f"Missing required fields: {missing}")

    try:
        validate_input(input_df)
        logging.info(f"Request ID {request_id}: Input validation passed")
    except ValueError as e:
        logging.error(f"Request ID {request_id}: Input validation failed: {e}")
        raise

    # Ensure correct dtypes
    for field in numerical_features + ['accommodates_bedrooms', 'min_nights_reviews']:
        if field in input_df.columns and not np.issubdtype(input_df[field].dtype, np.number):
            logging.error(f"Request ID {request_id}: Field '{field}' must be numeric")
            raise TypeError(f"Field '{field}' must be numeric")

    for field in binary_features:
        if field in input_df.columns:
            input_df[field] = input_df[field].fillna(0).astype(int)
        else:
            input_df[field] = 0

    # Clip numerical features to match training
    for col in numerical_features + ['accommodates_bedrooms', 'min_nights_reviews']:
        if col in input_df.columns:
            if col in ['accommodates', 'bedrooms', 'beds']:
                input_df[col] = np.clip(input_df[col], 0, 10)
            elif col == 'bathrooms':
                input_df[col] = np.clip(input_df[col], 0.5, 5)
            elif col == 'minimum_nights':
                input_df[col] = np.clip(input_df[col], 1, 30)
            elif col == 'maximum_nights':
                input_df[col] = np.clip(input_df[col], 1, 365)  # Tighter cap
            elif col == 'number_of_reviews':
                input_df[col] = np.clip(input_df[col], 0, 500)
            elif col in ['review_scores_rating', 'avg_review_score']:
                input_df[col] = np.clip(input_df[col], 0, 100)
            elif col == 'host_age_days':
                input_df[col] = np.clip(input_df[col], 0, 7300)
            elif col == 'days_since_last_review':
                input_df[col] = np.clip(input_df[col], 0, 3650)
            elif col == 'season':
                input_df[col] = np.clip(input_df[col], 0, 3)
            elif col == 'distance_to_center_km':
                input_df[col] = np.clip(input_df[col], 0, 20)
            elif col == 'accommodates_bedrooms':
                input_df[col] = np.clip(input_df[col], 0, 50)
            elif col == 'min_nights_reviews':
                input_df[col] = np.clip(input_df[col], 0, 5000)

    # Log input feature stats
    numerical_stats = {col: input_df[col].iloc[0] for col in numerical_features + ['accommodates_bedrooms', 'min_nights_reviews'] if col in input_df.columns}
    logging.info(f"Request ID {request_id}: Input numerical features: {numerical_stats}")

    # Add dummy columns for neighbourhood_cleansed and room_type
    for col in model_columns:
        if col not in input_df.columns:
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

    # Preprocess and restore feature names
    try:
        processed = pipeline.transform(input_df)
        processed_df = pd.DataFrame(processed, columns=model_columns)
        if processed_df.shape[1] != 53:
            logging.error(f"Request ID {request_id}: Processed data has {processed_df.shape[1]} features, expected 53")
            raise ValueError(f"Processed data has {processed_df.shape[1]} features, expected 53")
        logging.info(f"Request ID {request_id}: Processed shape: {processed_df.shape}")
        logging.info(f"Request ID {request_id}: Processed feature stats: min={processed_df.min().min():.2f}, max={processed_df.max().max():.2f}")
        processed_df.to_csv(project_root / "logs" / f"processed_input_{request_id}.csv")
        logging.info(f"Request ID {request_id}: Saved processed input to logs/processed_input_{request_id}.csv")
    except Exception as e:
        logging.error(f"Request ID {request_id}: Pipeline transformation failed: {e}")
        raise

    # Predict with each model
    best_rf = ensemble_config['rf_model']
    best_ridge = ensemble_config['ridge_model']
    best_xgb = ensemble_config['xgboost_model']
    weights = ensemble_config['weights']

    y_pred_rf_log = best_rf.predict(processed_df)
    y_pred_rf_log = np.clip(y_pred_rf_log, np.log1p(15), np.log1p(1000))
    y_pred_rf = np.expm1(y_pred_rf_log)
    y_pred_rf = np.clip(y_pred_rf, 15, 1000)
    y_pred_ridge_log = best_ridge.predict(processed_df)
    y_pred_ridge_log = np.clip(y_pred_ridge_log, np.log1p(15), np.log1p(1000))
    y_pred_ridge = np.expm1(y_pred_ridge_log)
    y_pred_ridge = np.clip(y_pred_ridge, 15, 1000)
    y_pred_xgb_log = best_xgb.predict(processed_df)
    y_pred_xgb_log = np.clip(y_pred_xgb_log, np.log1p(15), np.log1p(1000))
    y_pred_xgb = np.expm1(y_pred_xgb_log)
    y_pred_xgb = np.clip(y_pred_xgb, 15, 1000)

    # Log individual predictions
    logging.info(f"Request ID {request_id}: RF log={y_pred_rf_log[0]:.2f}, Ridge log={y_pred_ridge_log[0]:.2f}, XGB log={y_pred_xgb_log[0]:.2f}")
    logging.info(f"Request ID {request_id}: RF={y_pred_rf[0]:.2f}, Ridge={y_pred_ridge[0]:.2f}, XGB={y_pred_xgb[0]:.2f}")

    # Combine predictions
    prediction = weights['rf'] * y_pred_rf + weights['ridge'] * y_pred_ridge + weights['xgboost'] * y_pred_xgb
    prediction = np.clip(prediction, 15, 1000)
    prediction = round(float(prediction[0]), 2)

    # Log to file
    input_str = ", ".join(f"{k}={v}" for k, v in input_data.items())
    log_msg = (
        f"Request ID: {request_id}\n"
        f"Timestamp: {timestamp}\n"
        f"Input: {input_str}\n"
        f"Prediction: ${prediction:.2f} CAD\n"
    )
    logging.info(log_msg)

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
        predict_price(listing_data)
    except Exception as e:
        logging.error(f"Main: Failed to process input JSON: {e}")
        print(f"Failed to process input JSON: {e}")
        sys.exit(1)