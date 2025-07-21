import joblib

# Load the saved model columns
model_columns = joblib.load("models/model_columns.joblib")

# Write them using UTF-8 encoding
with open("model_columns_output.txt", "w", encoding="utf-8") as f:
    for col in model_columns:
        f.write(f"{col}\n")

print("Model columns saved to model_columns_output.txt using UTF-8 encoding.")
