from __future__ import annotations

import pandas as pd

# Load the CSV file
df = pd.read_csv("10-EVALUATION/models_ffs/model_ffs_eval_T5.csv")
# Create a list from the ModelID column
model_ids = df["ModelID"].tolist()



import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_json_files(folder):
    records = []
    folder = Path(folder)

    for path in folder.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                records.append(data)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    return records


def normalize_feature_item(item):
    """Make items hashable for uniqueness counting."""
    if isinstance(item, (str, int, float, bool)) or item is None:
        return str(item)
    return json.dumps(item, sort_keys=True)


def compute_stats(records, sample_model_ids=None):
    pipeline_counter = Counter()
    items_per_model = {}
    all_feature_items = []

    for record in records:
        model_id = record.get("modelID")
        if model_id is None:
            continue

        model_id = str(model_id)

        if sample_model_ids is not None and model_id not in sample_model_ids:
            continue

        pipeline_tag = record.get("Metadata", {}).get("pipeline_tag", "MISSING")
        pipeline_counter[str(pipeline_tag)] += 1

        features = record.get("Features", [])
        if not isinstance(features, list):
            features = []

        items_per_model[model_id] = len(features)
        all_feature_items.extend(normalize_feature_item(x) for x in features)

    counts = list(items_per_model.values())

    return {
        "n_models": len(items_per_model),
        "pipeline_tag_distribution": dict(pipeline_counter),
        "total_feature_items": len(all_feature_items),
        "total_unique_feature_items": len(set(all_feature_items)),
        "mean_items_per_model": statistics.mean(counts) if counts else 0,
        "median_items_per_model": statistics.median(counts) if counts else 0,
        "items_per_model": items_per_model,
    }


# -------------------------
# Example usage
# -------------------------

json_folder = "HF-Models-T7-U"

sample_model_ids = model_ids

records = load_json_files(json_folder)

all_stats = compute_stats(records)
sample_stats = compute_stats(records, sample_model_ids)

print("\nALL MODELS")
for k, v in all_stats.items():
    print(k, ":", v)

print("\nSAMPLE MODELS")
for k, v in sample_stats.items():
    print(k, ":", v)

def print_top_models_by_features(stats, title, top_n=50):
    """
    Print models with the most features.
    """
    items_per_model = stats["items_per_model"]

    sorted_models = sorted(
        items_per_model.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    print(f"\nTop {top_n} models with most features — {title}")
    print("-" * 50)

    for rank, (model_id, count) in enumerate(sorted_models, 1):
        print(f"{rank:2d}. {model_id} : {count} features")

print_top_models_by_features(all_stats, "ALL MODELS")
print_top_models_by_features(sample_stats, "SAMPLE MODELS")

# ---------------------------------------------------------------------
# # PIE CHART OF PIPELINE TAG DISTRIBUTION FOR ALL AND SAMPLE MODELS

# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch


# def save_pipeline_tag_piecharts(all_stats, sample_stats, output_png="13-GRAPHS/Coverage_FF/pipeline_tag_distributions.png"):

#     all_dist = all_stats["pipeline_tag_distribution"]
#     sample_dist = sample_stats["pipeline_tag_distribution"]

#     # All pipeline tags
#     all_tags = sorted(set(all_dist) | set(sample_dist))

#     # Top 3 tags from ALL models
#     top3 = sorted(all_dist.items(), key=lambda x: x[1], reverse=True)[:3]
#     top3_tags = [t[0] for t in top3]

#     # Nice colors for the largest slices
#     pretty_colors = ["#05478DFF", "#B64208", "#A415A7"]

#     # Generate enough unique colors for the rest
#     cmap = plt.get_cmap("tab20")
#     cmap2 = plt.get_cmap("tab20b")
#     cmap3 = plt.get_cmap("tab20c")

#     palette = [cmap(i) for i in range(20)] + \
#               [cmap2(i) for i in range(20)] + \
#               [cmap3(i) for i in range(20)]

#     color_map = {}

#     palette_i = 0
#     for tag in all_tags:
#         if tag in top3_tags:
#             color_map[tag] = pretty_colors[top3_tags.index(tag)]
#         else:
#             color_map[tag] = palette[palette_i]
#             palette_i += 1

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     def draw_pie(ax, dist, title):

#         labels = list(dist.keys())
#         sizes = list(dist.values())
#         colors = [color_map[l] for l in labels]

#         ax.pie(
#             sizes,
#             colors=colors,
#             startangle=90
#         )

#         ax.set_title(title)
#         ax.axis("equal")

#     draw_pie(axes[0], all_dist, "All Models")
#     draw_pie(axes[1], sample_dist, "Sample Models")

#     # Single legend for top 3
#     legend_patches = [
#         Patch(color=color_map[tag], label=f"{tag} ({count})")
#         for tag, count in top3
#     ]

#     fig.legend(
#     handles=legend_patches,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 0.95),
#     title="Top pipeline tags",
#     frameon=False
# )

#     plt.tight_layout()
#     plt.savefig(output_png, dpi=100, bbox_inches="tight")
#     plt.close()

#     print(f"Saved pie chart to: {output_png}")

# # Example usage:
# # save_pipeline_tag_piecharts(all_stats, sample_stats)
# save_pipeline_tag_piecharts(all_stats, sample_stats)