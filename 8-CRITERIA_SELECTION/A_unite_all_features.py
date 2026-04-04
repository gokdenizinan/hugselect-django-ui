import json
import os
from collections import defaultdict
import re
import pandas as pd

print("Starting to unite all features...")
# === LOAD DATA ===
with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different.json", "r", encoding="utf-8") as f:
    metadata_2 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_2.json", "r", encoding="utf-8") as f:
    metadata_3 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_all_stats_different_3.json", "r", encoding="utf-8") as f:
    metadata_4 = json.load(f)

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_fuzzy_full.json", "r", encoding="utf-8") as f:
    quality = json.load(f)

functional = pd.read_csv("4-LLM_FEATURE_ORGANIZATION/df_model_info.csv") 
# === CHECK DATA ===
metadata.extend(metadata_2)
metadata.extend(metadata_3)
metadata.extend(metadata_4)

# ids1 = {v["model_id"] for v in model_dict.values()}
# ids2 = {d["model_id"] for d in metadata}
# print(len(ids1))
# print(len(ids2))

# if ids1 == ids2:
#     print("Both have the same model_ids!")
# else:
#     print("They differ.")
#     print("Only in dict1:", len(ids1 - ids2))
#     print("Only in list2:", len(ids2 - ids1))


# === HELPER FUNCTIONS ===

input_folder = "/Users/ardacanseradali/Documents/Thesis_master/HF-Models-T1"
output_folder = "/Users/ardacanseradali/Documents/Thesis_master/HF-Models-T4"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load your list of dictionaries
# Build a lookup from the folder dictionaries
# Starting from your dict-of-dicts
# I'll call it `models_info` (change the variable name to your actual one)

name_to_ids = {}

for entry in model_dict.values():
    model_id = entry["model_id"]
    model_name = entry["model_name"]   # adjust if column name differs
    name_to_ids.setdefault(model_name, []).append(model_id)


# Create lookup for list_of_dicts
lookup = {d["model_id"]: d for d in metadata}
features_by_name = (
    functional.set_index("Model Name")["Features"]
      .apply(lambda x: eval(x) if isinstance(x, str) else x)
      .to_dict()
)

features_by_id = {}

for name, ids in name_to_ids.items():
    if name not in features_by_name:
        continue  # no features for this name, skip

    feats = features_by_name[name]

    for mid in ids:
        features_by_id[mid] = feats

fuzzy_lookup = {d["model_id"]: d for d in quality}



for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load JSON
        with open(input_path, "r") as f:
            data = json.load(f)

        model_id = data.get("modelId")
        

        # Add Features if the model_id exists in the DataFrame
        if model_id in lookup:
            data.update(lookup[model_id])
        
        original_keys = set(data.keys())
        if model_id in features_by_id:
            data["Features"] = features_by_id[model_id]

        # 3) add fuzzy_score + num_LH as flat keys
        if model_id in fuzzy_lookup:
            fuzzy_entry = fuzzy_lookup[model_id]

            for key, value in fuzzy_entry.items():
                if key == "model_id":
                    continue
                if not isinstance(value, dict):
                    continue

                fuzzy = value.get("fuzzy_score")
                num_lh = value.get("num_LH")

                # Only add if fuzzy_score has a value
                if fuzzy is not None:
                    data[key] = { "score":fuzzy, "num_LH": num_lh }


        nested = {}

        for key in list(original_keys):  # iterate over original keys only
            # keep some keys at top level
            if key in {"model_id", "modelId", "author", "Features", "description"}:
                continue
            if "fuzzy" in key:  # skip anything whose name contains "fuzzy"
                continue
            if key not in data:
                # might have been removed or changed already
                continue

            # move this original key into "original_data"
            nested[key] = data[key]
            del data[key]

        if nested:
            data["metadata"] = nested  # rename this if you want

        data.pop("model_id", None)

        # Write updated version into output folder
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

print("Done! Updated JSON files saved to:", output_folder)

