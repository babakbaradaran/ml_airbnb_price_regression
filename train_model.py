import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from src.preprocessing import build_preprocessing_pipeline
from sklearn.model_selection import GridSearchCV

# Set paths
data_path = Path("data/processed/featured_listings.csv")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Load training data
df = pd.read_csv(data_path)
df = df.dropna(subset=["price"])  # Drop listings with missing target

# Separate target and features
y = df["price"]
X = df.drop(columns=["price"])

# Build and fit preprocessing pipeline
pipeline, numerical_features, categorical_features = build_preprocessing_pipeline()
pipeline.fit(X)

# Transform features using the fitted pipeline
X_processed = pipeline.transform(X)

# Convert to DataFrame (handling sparse output if needed)
X_processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

xgb_model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_processed_df, y)
best_xgb = grid_search.best_estimator_

# Save model and artifacts
joblib.dump(best_xgb, model_dir / "xgboost_model.joblib")
joblib.dump(X.columns.tolist(), model_dir / "model_columns.joblib")
joblib.dump(pipeline, model_dir / "fitted_pipeline.joblib")

print("Best XGBoost model, pipeline, and feature columns saved.")
