from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import pandas as pd
import numpy as np

def identity_function(X):
    """Pass-through function for dummy columns."""
    return X

def build_preprocessing_pipeline():
    """
    Builds a preprocessing pipeline for Airbnb price prediction, including feature engineering and scaling.

    Returns:
    - pipeline: ColumnTransformer pipeline
    - numerical_features: List of numerical columns used
    - binary_features: List of binary columns used
    - dummy_features: List of dummy columns (neighbourhood_cleansed_*, room_*)
    """
    numerical_features = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights',
        'number_of_reviews', 'review_scores_rating',
        'host_age_days', 'days_since_last_review', 'avg_review_score',
        'accommodates_bedrooms', 'min_nights_reviews', 'season', 'distance_to_center_km'
    ]

    binary_features = [
        'Has_Wifi', 'Has_Kitchen', 'Has_Heating', 'Has_TV', 'Has_Essentials',
        'Has_Hair_Dryer', 'Has_Iron', 'Has_Free_Parking', 'Has_Hangers',
        'Has_Laptop_Friendly_Workspace', 'host_has_profile_pic_binary',
        'host_identity_verified_binary', 'host_is_superhost', 'instant_bookable'
    ]

    dummy_features = []  # Populated dynamically during fitting

    # Numerical transformer: Impute with mean, then scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Binary transformer: Impute with 0
    binary_transformer = SimpleImputer(strategy='constant', fill_value=0)

    # Dummy transformer: Pass through pre-encoded dummy columns
    dummy_transformer = FunctionTransformer(identity_function, validate=False)

    # Combine transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bin', binary_transformer, binary_features),
        ('dum', dummy_transformer, dummy_features)
    ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline, numerical_features, binary_features, dummy_features

def compute_interaction_terms(X):
    """Compute interaction terms for the dataset."""
    X = X.copy()
    if 'accommodates' in X.columns and 'bedrooms' in X.columns:
        X['accommodates_bedrooms'] = X['accommodates'] * X['bedrooms']
    if 'minimum_nights' in X.columns and 'number_of_reviews' in X.columns:
        X['min_nights_reviews'] = X['minimum_nights'] * X['number_of_reviews']
    return X