import json
import csv
from pathlib import Path

# --- paths ---
DICT_FOLDER = Path("HF-Models-T7-U")
CSV_PATH = Path("7-CLUSTERING_MODELS/clusters_improved/family_assignments_organized.csv")

# Change these if needed
DICT_KEY = "modelID"
CSV_KEY = "model_id"

# Use the exact CSV column names here
CLUSTER_COLUMNS = [
    "assigned_modality",
    "task",
    "family_root",
    "family_child",
]

# If your CSV really has "task family_root" as one column by mistake,
# replace the list above with:
# CLUSTER_COLUMNS = ["assigned_modality", "task family_root", "family_child"]


def load_csv_lookup(csv_path: Path) -> dict:
    lookup = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = str(row[CSV_KEY]).strip()
            lookup[model_id] = {
                col: row.get(col)
                for col in CLUSTER_COLUMNS
                if col in row
            }
    return lookup


def add_clusters_to_dict(data, lookup):
    """
    Handles:
    - a single dictionary
    - a list of dictionaries
    """
    if isinstance(data, dict):
        model_id = str(data.get(DICT_KEY, "")).strip()
        if model_id in lookup:
            data["Clusters"] = lookup[model_id]
        return data

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                model_id = str(item.get(DICT_KEY, "")).strip()
                if model_id in lookup:
                    item["Clusters"] = lookup[model_id]
        return data

    return data


def main():
    lookup = load_csv_lookup(CSV_PATH)

    json_files = list(DICT_FOLDER.glob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        updated_data = add_clusters_to_dict(data, lookup)

        with json_file.open("w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)

        print(f"Updated: {json_file.name}")


if __name__ == "__main__":
    main()