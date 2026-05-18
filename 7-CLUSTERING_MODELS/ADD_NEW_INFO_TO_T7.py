import json
import csv
from pathlib import Path

# --- paths ---
DICT_FOLDER = Path("HF-Models-T7-U")
CSV_PATH = Path("7-CLUSTERING_MODELS/clusters_improved_2/family_assignments_organized.csv")

# Change these if needed
DICT_KEY = "modelID"
CSV_KEY = "model_id"

# Use the exact CSV column names here
CLUSTER_COLUMNS = [
    "assigned_domain",
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

    Logic:
    - Remove existing "Clusters"
    - Add new one only if lookup match exists
    """

    def process_item(item):
        if not isinstance(item, dict):
            return item

        model_id = str(item.get(DICT_KEY, "")).strip()

        existing_clusters = item.get("Clusters", {})

        if not isinstance(existing_clusters, dict):
            existing_clusters = {}

        # Keep task exactly as it already is
        existing_task = existing_clusters.get("task")

        # Remove modality from existing Clusters
        existing_clusters.pop("assigned_modality", None)

        if model_id in lookup and lookup[model_id]:
            new_data = lookup[model_id].copy()

            # Rename assigned_domain -> domain
            if "assigned_domain" in new_data:
                new_data["domain"] = new_data.pop("assigned_domain")

            # Do not overwrite task
            if existing_task is not None:
                new_data["task"] = existing_task

            item["Clusters"] = {**existing_clusters, **new_data}
        else:
            item["Clusters"] = existing_clusters

        return item
    
        # ✅ APPLY FUNCTION
    if isinstance(data, list):
        return [process_item(item) for item in data]
    elif isinstance(data, dict):
        return process_item(data)
    else:
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