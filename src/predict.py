import joblib
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to sys.path so imports like `from src...` work
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import build_preprocessing_pipeline

# ------------------- Setup Logging -------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename=log_dir / "prediction.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------- Load Model & Preprocessor -------------------
try:
    model = joblib.load("models/random_forest_model.joblib")
    model_columns = joblib.load("models/model_columns.joblib")
    pipeline, numerical_features, categorical_features = build_preprocessing_pipeline()
except Exception as e:
    logging.exception("Failed to load model or preprocessing pipeline.")
    raise e

# ------------------- Prediction Function -------------------
def predict_price_advanced_module(input_data):
    """
    Enhanced price prediction with:
    - Input validation
    - Type checking
    - Scikit-learn preprocessing pipeline
    - Error handling and logging
    - Modular preprocessing via src.preprocessing
    """
    try:
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            input_df = pd.DataFrame([input_data.to_dict()])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input must be a dict, pd.Series, or pd.DataFrame")

        # Log the input listing details
        logging.info(f"Input listing: {input_df.to_dict(orient='records')[0]}")

        # Validate required fields
        required_fields = numerical_features + categorical_features
        missing_fields = [f for f in required_fields if f not in input_df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Type checking for numerical fields
        for field in numerical_features:
            if not np.issubdtype(input_df[field].dtype, np.number):
                raise TypeError(f"Field '{field}' must be numeric")

        logging.info("Input validation passed")

        # Preprocessing
        logging.info("Preprocessing input for prediction")
        processed_input = pipeline.fit_transform(input_df)

        # Build encoded DataFrame and align with training columns
        encoded_df = pd.DataFrame(
            processed_input.toarray() if hasattr(processed_input, 'toarray') else processed_input
        )
        encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(encoded_df)[0]
        prediction = round(float(prediction), 2)

        # Log the prediction result
        logging.info(f"Prediction completed successfully. Predicted price: ${prediction:.2f} CAD")

        # Print explanation
        print("\nHow this works:")
        print("- Input is validated and converted to DataFrame.")
        print("- Numerical fields are imputed using means.")
        print("- Categorical fields are one-hot encoded.")
        print("- Result is aligned with model training features.")
        print("- Prediction is made using the trained model.\n")
        print(f"Predicted price for this listing: ${prediction:.2f} CAD")
        print("This is the estimated price per night (in CAD) for the listing described above.")

        return prediction

    except Exception as e:
        logging.exception("Prediction failed.")
        raise e

# ------------------- CLI Interface -------------------
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Airbnb listing price from input JSON")
    parser.add_argument("input_json", help="Path to JSON file containing listing data")
    args = parser.parse_args()

    try:
        logging.info(f"Loading input from: {args.input_json}")
        with open(args.input_json, "r") as f:
            listing_data = json.load(f)
    except Exception as e:
        logging.exception("Failed to read input JSON file.")
        print(f"Failed to read JSON file: {e}")
        sys.exit(1)

    try:
        predicted = predict_price_advanced_module(listing_data)
        print(f"\nPredicted price: ${predicted:.2f} CAD")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        print(f"Prediction failed: {e}")
        sys.exit(2)