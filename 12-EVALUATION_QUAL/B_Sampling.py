import random
import json

with open('6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json', 'r') as f:
    data = json.load(f)

def sample_and_rename(data: dict, n: int = 500, seed: int | None = None) -> dict:
    rng = random.Random(seed)

    keys = list(data.keys())
    if n > len(keys):
        n = len(keys)

    sampled_keys = rng.sample(keys, n)

    # map "old field name" -> "new field name"
    rename_map = {
        "mode_2": "old_mode_2",
        "model_used": "old_model_used",
        "Primary_Category": "old_primary_category",
        "Secondary_Categories": "old_secondary_category",
        "Rationale": "old_rationale",
        "Sentiment": "old_sentiment",
        "output": "old_output",
    }

    new_data = {}
    for k in sampled_keys:
        row = dict(data[k])  # shallow copy

        # rename if present
        for old_key, new_key in rename_map.items():
            if old_key in row:
                row[new_key] = row.pop(old_key)

        new_data[k] = row

    return new_data

# usage:
new_dict = sample_and_rename(data, n=500, seed=42)  # seed optional for reproducibility

# sort by numeric value of the outer key
# sort by original numeric key
sorted_items = sorted(new_dict.items(), key=lambda x: int(x[0]))

# reassign keys from 0..N-1
reindexed_dict = {
    str(i): value
    for i, (_, value) in enumerate(sorted_items)
}


with open("10-EVALUATION/llm_ffs/input_B1.json", "w", encoding="utf-8") as f:
    json.dump(reindexed_dict, f, ensure_ascii=False, indent=2)