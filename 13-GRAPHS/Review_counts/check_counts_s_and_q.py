#!/usr/bin/env python3

from pathlib import Path
import json
import pandas as pd


INPUT_FOLDER = Path("united_f5 copy")
MODEL_DICT_PATH = Path("1-MODEL_FILTERING/model_likes_10k_Y3.json")
OUTPUT_CSV = Path("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/source_stats_only.csv")

# united_f5 key -> output/stat name
SOURCE_KEYS = {
    "reddit": "reddit",
    "hf": "huggingface",
    "stack": "stack",
}


def load_json_files(folder_path: Path):
    """Load all JSON files from united_f5 folder."""
    items = []

    for file_path in folder_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                items.extend(data)
            elif isinstance(data, dict):
                items.append(data)

        except Exception as e:
            print(f"Warning: could not read {file_path}: {e}")

    return items


def load_first_100_model_keys(json_path: Path):
    """Load first 100 keys from model_likes_10k_Y3.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        model_dict = json.load(f)

    if not isinstance(model_dict, dict):
        raise ValueError("model_likes_10k_Y3.json must be a dictionary.")

    return set(list(model_dict.keys())[:100])


def get_source_stats(model_entries, top_100_model_ids):
    rows = []

    for raw_key, source_name in SOURCE_KEYS.items():
        row_count = 0
        model_ids_for_source = set()

        for entry in model_entries:
            model_id = entry.get("model_id", "")
            source_reviews = entry.get(raw_key, []) or []

            for review in source_reviews:
                mentioned_items = review.get("mentioned", []) or []

                for text in mentioned_items:
                    if isinstance(text, str) and text.strip():
                        row_count += 1
                        model_ids_for_source.add(model_id)

        top_100_matches = model_ids_for_source & top_100_model_ids

        rows.append({
            "source": source_name,
            "num_rows": row_count,
            "num_unique_models": len(model_ids_for_source),
            "num_models_in_first_100_model_likes_dict": len(top_100_matches),
            "matched_model_ids_in_first_100": "; ".join(sorted(top_100_matches)),
        })

    return pd.DataFrame(rows)


def main():
    model_entries = load_json_files(INPUT_FOLDER)
    top_100_model_ids = load_first_100_model_keys(MODEL_DICT_PATH)

    df_stats = get_source_stats(model_entries, top_100_model_ids)

    # OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # df_stats.to_csv(OUTPUT_CSV, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    df_stats = df_stats.drop(columns=["matched_model_ids_in_first_100"], errors="ignore")
    print(df_stats)
    print(f"\nSaved stats to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()