from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def build_preprocessing_pipeline():
    """
    Builds and returns a preprocessing pipeline for Airbnb price prediction.

    Returns:
    - pipeline: ColumnTransformer pipeline
    - numerical_features: List of numerical columns used
    - categorical_features: List of categorical columns used
    """
    numerical_features = [
        'accommodates',
        'bathrooms',
        'bedrooms',
        'beds',
        'minimum_nights',
        'maximum_nights',
        'number_of_reviews', 
        'review_scores_rating'
    ]

    categorical_features = [
        'host_is_superhost',
        'room_type',
        'instant_bookable',
        'neighbourhood_cleansed'
    ]

    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline, numerical_features, categorical_features
