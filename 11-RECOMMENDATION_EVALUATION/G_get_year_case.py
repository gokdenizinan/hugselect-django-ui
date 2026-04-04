import os
import json
import pandas as pd
from datetime import datetime

# --- paths ---
csv_path = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/snipped_papers_7.csv" 
json_folder = "HF-Models-T7-U"
output_path = "11-RECOMMENDATION_EVALUATION/model_to_year_more.json"

# --- read CSV robustly ---
try:
    df = pd.read_csv(csv_path, sep=None, engine="python")
except Exception:
    df = pd.read_csv(csv_path, sep=";", engine="python")

print("Columns found:", df.columns.tolist())

# --- get target model ids ---
target_models = set(df["matched_hf_model"].dropna().astype(str).str.strip().unique())

# --- build mapping ---
model_to_year = {}

for filename in os.listdir(json_folder):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(json_folder, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_id = str(data.get("modelID", "")).strip()
        last_modified = data.get("Metadata", {}).get("lastModified")

        if model_id in target_models and last_modified:
            year = datetime.fromisoformat(last_modified).year
            model_to_year[model_id] = year

    except Exception as e:
        print(f"Skipping {filename}: {e}")

# --- save output ---
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(model_to_year, f, indent=2, ensure_ascii=False)

print(f"Saved {len(model_to_year)} entries to {output_path}")