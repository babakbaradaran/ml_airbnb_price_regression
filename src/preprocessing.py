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
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights',
        'number_of_reviews', 'review_scores_rating',
        'availability_30', 'availability_60', 'availability_90', 'availability_365',
        'host_listings_count', 'host_total_listings_count'
    ]

    categorical_features = [
        'host_is_superhost', 'instant_bookable', 'room_type', 'neighbourhood_cleansed'
    ]

    # Impute numericals with mean, categoricals with most frequent
    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline, numerical_features, categorical_features
