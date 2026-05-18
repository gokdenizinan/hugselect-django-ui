from pathlib import Path
import json
from collections import defaultdict

REVIEWS_DIR = Path("united_f7 copy")
MODEL_LIKES_PATH = Path("1-MODEL_FILTERING/model_likes_10k_Y3.json")
from pathlib import Path
import json


SOURCE_KEYS = ["reddit", "hf", "stack"]

# load model_likes and take first 100
with open(MODEL_LIKES_PATH, "r", encoding="utf-8") as f:
    model_likes_10k_Y3 = json.load(f)

first_100_models = set(list(model_likes_10k_Y3.keys())[:100])

stats = {
    key: {
        "models_with_values": set(),
        "models_in_first_100": set(),
        "individual_mentioned_reviews": 0,
    }
    for key in SOURCE_KEYS
}

# ---- main loop ----
for path in REVIEWS_DIR.glob("*.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_id = data.get("model_id")
    if not model_id:
        continue

    for key in SOURCE_KEYS:
        values = data.get(key)

        if values:
            stats[key]["models_with_values"].add(model_id)

            if model_id in first_100_models:
                stats[key]["models_in_first_100"].add(model_id)

            for item in values:
                mentioned = item.get("mentioned", [])
                if isinstance(mentioned, list):
                    stats[key]["individual_mentioned_reviews"] += len(mentioned)
                elif mentioned:
                    stats[key]["individual_mentioned_reviews"] += 1

# ---- per-source summary ----
summary = {}

all_models_with_values = set()
all_models_in_first_100 = set()
total_mentions = 0

for key, s in stats.items():
    summary[key] = {
        "num_models_with_values": len(s["models_with_values"]),
        "num_models_in_first_100": len(s["models_in_first_100"]),
        "num_individual_mentioned_reviews": s["individual_mentioned_reviews"],
    }

    # accumulate totals
    all_models_with_values |= s["models_with_values"]
    all_models_in_first_100 |= s["models_in_first_100"]
    total_mentions += s["individual_mentioned_reviews"]

# ---- total block ----
summary["total"] = {
    "num_models_with_values": len(all_models_with_values),
    "num_models_in_first_100": len(all_models_in_first_100),
    "num_individual_mentioned_reviews": total_mentions,
}

print(json.dumps(summary, indent=2))