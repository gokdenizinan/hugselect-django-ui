import os
import json
import csv
from pathlib import Path

INPUT_FOLDER = "HF-Models-T7-U"
OUTPUT_CSV = "7-CLUSTERING_MODELS/hf_models.csv"


def extract_model_name(model_id: str) -> str:
    if not model_id:
        return ""
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def safe_join(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return str(value)


def normalize_record(record: dict) -> dict:
    model_id = record.get("modelID", "")
    description = record.get("description", "") or ""
    metadata = record.get("Metadata", {}) or {}

    return {
        "model_id": model_id,
        "model_name": extract_model_name(model_id),
        "description": description,
        "short_description": description[:500],
        "tags": safe_join(metadata.get("tags")),
        "pipeline_tag": metadata.get("pipeline_tag", ""),
        "model_type": metadata.get("model_type", ""),
        "base_models": safe_join(metadata.get("basemodels")),
        "library_name": metadata.get("library_name", ""),
    }


def load_json_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either:
    # 1. one object per file
    # 2. a list of objects per file
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        return []


def main():
    input_path = Path(INPUT_FOLDER)
    rows = []

    for file_path in input_path.glob("*.json"):
        try:
            records = load_json_file(file_path)
            for record in records:
                if isinstance(record, dict):
                    rows.append(normalize_record(record))
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    fieldnames = [
        "model_id",
        "model_name",
        "description",
        "short_description",
        "tags",
        "pipeline_tag",
        "model_type",
        "base_models",
        "library_name",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()