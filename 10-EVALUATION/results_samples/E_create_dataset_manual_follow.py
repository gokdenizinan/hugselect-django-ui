import pandas as pd
import json

# Load CSV
df = pd.read_csv("10-EVALUATION/llm_ffs/manuel_gemini.csv")

# Function to extract features list
def extract_features(truth_value):
    if pd.isna(truth_value):
        return []
    try:
        parsed = json.loads(truth_value)
        return parsed.get("features", [])
    except json.JSONDecodeError:
        return []

# Apply to column
df["features_list"] = df["truth"].apply(extract_features)

print(df["features_list"].iloc[0])