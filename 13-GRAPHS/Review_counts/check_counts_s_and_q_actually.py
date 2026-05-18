import json
from pathlib import Path
from collections import Counter, defaultdict

quality_path = Path("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json")
likes_path = Path("1-MODEL_FILTERING/model_likes_10k_Y3.json")

with open(quality_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(likes_path, "r", encoding="utf-8") as f:
    likes_data = json.load(f)


def extract_model_ids(obj):
    """
    Handles common formats:
    - dict where keys are model_ids
    - dict where values contain model_id
    - list of model_ids
    - list of dicts containing model_id
    """
    model_ids = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and "model_id" in v:
                model_ids.append(v["model_id"])
            else:
                model_ids.append(k)

    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict) and "model_id" in x:
                model_ids.append(x["model_id"])
            elif isinstance(x, str):
                model_ids.append(x)

    return model_ids


top_100_models = set(extract_model_ids(likes_data)[:100])


def summarize_group(items):
    models = {
        item.get("model_id")
        for item in items
        if item.get("model_id")
    }

    return {
        "total_count": len(items),
        "total_unique_models": len(models),
        "unique_models_in_top_100": len(models & top_100_models),
    }


all_items = list(data.values())

results = {}

# Overall
results["overall"] = summarize_group(all_items)

# Sentiment stats
for sentiment in ["Positive", "Neutral", "Negative"]:
    group = [
        item for item in all_items
        if item.get("Sentiment") == sentiment
    ]
    results[f"Sentiment = {sentiment}"] = summarize_group(group)

# Primary category unclear / not unclear
unclear_items = [
    item for item in all_items
    if item.get("Primary_Category") == "Unclear"
]

not_unclear_items = [
    item for item in all_items
    if item.get("Primary_Category") != "Unclear"
]

results["Primary_Category = Unclear"] = summarize_group(unclear_items)
results["Primary_Category != Unclear"] = summarize_group(not_unclear_items)

# mode_2 stats
mode_values = sorted(
    {item.get("mode_2") for item in all_items},
    key=lambda x: (str(type(x)), str(x))
)

for mode in mode_values:
    group = [
        item for item in all_items
        if item.get("mode_2") == mode
    ]
    results[f"mode_2 = {mode}"] = summarize_group(group)


# Print nicely
print("\n=== STATS ===")
for group_name, stats in results.items():
    print(f"\n{group_name}")
    print(f"  Total count: {stats['total_count']}")
    print(f"  Total unique models: {stats['total_unique_models']}")
    print(f"  Unique models in top 100 model_likes_10k_Y3: {stats['unique_models_in_top_100']}")