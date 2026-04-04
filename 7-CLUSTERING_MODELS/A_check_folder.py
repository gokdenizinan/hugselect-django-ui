import os
import json

folder_path = "HF-Models-T7-U"

models = []

def has_real_family(x):
    if x is None:
        return False
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    return x not in {"", "Other / Unclear"}

for filename in os.listdir(folder_path):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(folder_path, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    name = data.get("modelID", filename)
    metadata = data.get("Metadata", {})
    clusters = data.get("Clusters", {})

    likes = metadata.get("likes", 0)
    model_type = metadata.get("model_type")
    family_root = clusters.get("family_root")

    models.append({
        "name": name,
        "likes": likes if isinstance(likes, (int, float)) else 0,
        "has_model_type": bool(model_type),
        "has_family" : has_real_family(family_root)
    })

# ---- sort by likes (descending) ----
models.sort(key=lambda x: x["likes"], reverse=True)

# ---- choose top N ----
TOP_N = 100  # change this
top_models = models[:TOP_N]

# ---- count missing model_type ----
missing_model_type = sum(1 for m in top_models if not m["has_model_type"])
missing_family_root = sum(1 for m in top_models if not m["has_family"])


print(f"Top {TOP_N} models:")
print(f"Missing model_type: {missing_model_type}")
print(f"Percentage: {missing_model_type / TOP_N:.2%}")
print("")

print(f"Missing family_root: {missing_family_root}")
print(f"Percentage: {missing_family_root / TOP_N:.2%}")


with open("7-CLUSTERING_MODELS/cluster_counts/pipeline_tag_counts.json","r") as f:
    model_type_counts = json.load(f)

total = sum(model_type_counts.values())
print("pipeline_tag_counts: ", total)  # 30

print("\nTop model names:")
for i, m in enumerate(top_models, 1):
    print(f"{i}. {m['name']} (likes: {m['likes']}) -  has_family: {m['has_family']}")