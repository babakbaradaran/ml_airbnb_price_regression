import joblib
from pathlib import Path

# Load old model and pipeline
model = joblib.load("models/random_forest_model.joblib")
pipeline = joblib.load("models/fitted_pipeline.joblib")
model_columns = joblib.load("models/model_columns.joblib")

# Re-save using current joblib/numpy versions
Path("resaved_models").mkdir(exist_ok=True)

joblib.dump(model, "resaved_models/random_forest_model.joblib")
joblib.dump(pipeline, "resaved_models/fitted_pipeline.joblib")
joblib.dump(model_columns, "resaved_models/model_columns.joblib")
