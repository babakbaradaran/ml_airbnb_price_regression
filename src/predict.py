import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained model and model columns
model = joblib.load("models/random_forest_model.joblib")
model_columns = joblib.load("models/model_columns.joblib")

# Define preprocessing pipeline
numerical_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
                      'maximum_nights', 'number_of_reviews', 'review_scores_rating']
categorical_features = ['host_is_superhost', 'room_type', 'instant_bookable', 'neighbourhood_cleansed']

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

def predict_price_advanced(input_data):
    """
    Enhanced price prediction with:
    - Input validation
    - Type checking
    - Scikit-learn preprocessing pipeline
    - Error handling and logging
    - Explanation of each step for clarity
    """
    # Convert to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise ValueError("Input must be a dict, pd.Series, or pd.DataFrame")

    # Validate required fields
    required_fields = numerical_features + categorical_features
    missing_fields = [f for f in required_fields if f not in input_df.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Type checking
    for field in numerical_features:
        if not np.issubdtype(input_df[field].dtype, np.number):
            raise TypeError(f"Field '{field}' must be numeric")

    # Preprocess
    logging.info("Preprocessing input...")
    processed_input = pipeline.fit_transform(input_df)

    # Align with training columns
    encoded_df = pd.DataFrame(
        processed_input.toarray() if hasattr(processed_input, 'toarray') else processed_input
    )
    encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(encoded_df)[0]
    prediction = round(float(prediction), 2)

    # Explanation
    print("\nHow this works:")
    print("- The input listing is validated and converted to a DataFrame.")
    print("- Numerical features are imputed (missing values filled with mean).")
    print("- Categorical variables are one-hot encoded using scikit-learn.")
    print("- The resulting input is aligned with the original training columns.")
    print("- Finally, the trained Random Forest model predicts the price.\n")
    print(f"Predicted price for this listing: ${prediction:.2f} CAD")
    print("This is the estimated price per night (in CAD) for the listing provided above.")

    return prediction
