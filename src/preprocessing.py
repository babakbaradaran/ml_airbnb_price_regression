# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def build_preprocessing_pipeline(scaler=RobustScaler()):
    """
    Build a preprocessing pipeline for numerical, binary, and dummy features.
    
    Parameters:
    scaler: Scaler object for numerical features (default: RobustScaler())
    
    Returns:
    pipeline: Preprocessing pipeline
    numerical_features: List of numerical feature names
    binary_features: List of binary feature names
    dummy_features: List of dummy feature names
    """
    numerical_features = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights',
        'number_of_reviews', 'review_scores_rating',
        'host_age_days', 'days_since_last_review', 'avg_review_score',
        'season', 'distance_to_center_km',
        'accommodates_bedrooms', 'min_nights_reviews'
    ]
    binary_features = [
        'Has_Wifi', 'Has_Kitchen', 'Has_Heating', 'Has_TV', 'Has_Essentials',
        'Has_Hair_Dryer', 'Has_Iron', 'Has_Free_Parking', 'Has_Hangers',
        'Has_Laptop_Friendly_Workspace', 'host_has_profile_pic_binary',
        'host_identity_verified_binary', 'host_is_superhost', 'instant_bookable'
    ]
    dummy_features = [
        col for col in pd.read_csv("data/processed/featured_listings.csv").columns 
        if col.startswith('neighbourhood_cleansed_') or col.startswith('room_')
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_features),
            ('bin', 'passthrough', binary_features),
            ('dum', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), dummy_features)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    return pipeline, numerical_features, binary_features, dummy_features

def compute_interaction_terms(df):
    """
    Compute interaction terms for specific feature pairs.
    
    Parameters:
    df: DataFrame with input features
    
    Returns:
    df: DataFrame with added interaction terms
    """
    df = df.copy()
    if 'accommodates' in df.columns and 'bedrooms' in df.columns:
        df['accommodates_bedrooms'] = df['accommodates'] * df['bedrooms']
    if 'minimum_nights' in df.columns and 'number_of_reviews' in df.columns:
        df['min_nights_reviews'] = df['minimum_nights'] * df['number_of_reviews']
    return df