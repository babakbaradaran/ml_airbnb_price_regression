import joblib
import pandas as pd
from pathlib import Path

# Define the models directory
models_dir = Path("models")

# Initialize a dictionary to store column data
column_data = {}

# Load and extract columns from model_columns_v1_tuned.joblib
try:
    model_columns = joblib.load(models_dir / "model_columns_v1_tuned.joblib")
    column_data["model_columns_v1_tuned"] = model_columns
except FileNotFoundError:
    print("File model_columns_v1_tuned.joblib not found in models directory.")
except Exception as e:
    print(f"Error loading model_columns_v1_tuned.joblib: {e}")

# Load and extract feature names from fitted_pipeline.joblib
try:
    pipeline = joblib.load(models_dir / "fitted_pipeline.joblib")
    pipeline_columns = pipeline.get_feature_names_out()
    column_data["fitted_pipeline_columns"] = pipeline_columns
except FileNotFoundError:
    print("File fitted_pipeline.joblib not found in models directory.")
except Exception as e:
    print(f"Error loading fitted_pipeline.joblib: {e}")

# Convert to DataFrame
if column_data:
    df = pd.DataFrame.from_dict(column_data, orient="index").T
    # Ensure all columns are filled (some might be shorter)
    df = df.fillna("N/A")
else:
    print("No column data to export.")
    df = pd.DataFrame()

# Export to Excel
output_file = "column_names_export.xlsx"
df.to_excel(output_file, index=False, engine="openpyxl")
print(f"Columns exported to {output_file}")