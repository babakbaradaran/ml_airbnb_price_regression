import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing import build_preprocessing_pipeline

# Set paths
data_path = Path("data/processed/featured_listings.csv")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Load and clean training data
df = pd.read_csv(data_path)
df = df.dropna(subset=["price"])  # Drop listings with missing price

# Separate target and features
y = df["price"]
X = df.drop(columns=["price"])

# Build and fit preprocessing pipeline
pipeline, num_feats, cat_feats = build_preprocessing_pipeline()
pipeline.fit(X)

# Transform features
X_processed = pipeline.transform(X)
X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_processed_df, y)

# Save artifacts
joblib.dump(model, model_dir / "random_forest_model.joblib")
joblib.dump(X_processed_df.columns.tolist(), model_dir / "model_columns.joblib")
joblib.dump(pipeline, model_dir / "fitted_pipeline.joblib")

print("Model, pipeline, and columns saved successfully.")
