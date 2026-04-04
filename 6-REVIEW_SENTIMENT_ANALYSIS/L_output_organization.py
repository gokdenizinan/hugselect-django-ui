import json
import json
from pathlib import Path
import re
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



from pathlib import Path
import json

def unite_outputs():
    base_dir = Path("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping")
    input_pattern = "quality_mapping_output_*50_*_U.json"
    unified_path = base_dir / "quality_mapping_output_AB50_all.json"

    # 1) Load existing unified file if it exists
    if unified_path.exists():
        with unified_path.open("r", encoding="utf-8") as f:
            unified = json.load(f)
    else:
        unified = {}

    # Current max numeric key (as int) so we can assign new ones
    if unified:
        max_id = max(int(k) for k in unified.keys())
    else:
        max_id = 0

    # 2) Loop over all numbered JSON files (deterministic order)
    for path in sorted(base_dir.glob(input_pattern)):
        if path == unified_path:
            continue  # just in case

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 3) Merge keys
        for k, v in data.items():
            # New key: just add
            if k not in unified:
                unified[k] = v
                # keep max_id in sync with the largest key we see
                try:
                    kid = int(k)
                    if kid > max_id:
                        max_id = kid
                except ValueError:
                    # if for some reason k isn't an int string, just ignore for max_id
                    pass
            else:
                # Same key exists – check value
                if unified[k] == v:
                    # True duplicate, ignore
                    continue
                else:
                    # Different value – assign a new numeric key
                    max_id += 1
                    new_key = str(max_id)
                    unified[new_key] = v

    # 4) Save unified file
    with unified_path.open("w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    print(f"Unified {len(unified)} entries into {unified_path}")


def exract_output():
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all.json") as f:
        a50_all = json.load(f)


    def merge_output_json(record):
        new_record = copy.deepcopy(record)
        output = record.get("output", "")

        # Match ```json ... ``` OR ``` ... ```
        match = re.search(
            r"```(?:json)?\s*([\s\S]*?)```",
            output,
            re.IGNORECASE
        )

        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                new_record.update(parsed)
            except json.JSONDecodeError:
                # JSON inside code block wasn’t valid – optionally log or ignore
                pass

        return new_record

    # Apply to every record in your dictionary
    a50_all_expanded = {}

    for key, record in a50_all.items():
        a50_all_expanded[key] = merge_output_json(record)

    # Keep all items except those with mode_2 == -1 AND Sentiment == "Neutral"
    a50_all_expanded_filtered = {
        key: val
        for key, val in a50_all_expanded.items()
        if not (val.get("mode_2") == -1 and val.get("Sentiment") == "Neutral")
    }


    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json", "w", encoding="utf-8") as f:
        json.dump(a50_all_expanded_filtered, f, indent=2, ensure_ascii=False)


    # number of dictionaries that have the key "Primary_Category"
    with_primary = sum("Primary_Category" in d for d in a50_all_expanded_filtered.values())

    # set of all unique keys that appear across any inner dictionary
    unique_keys = set().union(*(d.keys() for d in a50_all_expanded_filtered.values()))

    print("Total dictionaries:", len(a50_all_expanded))
    print("Total dictionaries filtered for neutral:", len(a50_all_expanded_filtered))

    print("With 'Primary_Category':", with_primary)
    print("Unique keys:", unique_keys)
    print("Number of unique keys:", len(unique_keys))

    for key, inner in a50_all_expanded_filtered.items():
        if "Primary_Category" not in inner:
            print(f"{key}")


### FUZZY AGGREGATION

# ---------------------------------------------
# Simplified Fuzzy Aggregation (Low & High only)
# ---------------------------------------------

def fuzzy_aggregation_analysis(min_count: int = 3, count_neutral_toward_min: bool = False):
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json") as f:
        a50_all = json.load(f)

    from collections import defaultdict

    # Step 1: group by model_id
    grouped = defaultdict(lambda: defaultdict(list))

    for record in a50_all.values():
        model_id = record.get("model_id")
        category = record.get("Primary_Category")

        # --- Fix mode_2 if it is -1 based on sentiment ---
        if record.get("mode_2") == -1:
            sentiment = record.get("Sentiment", "").strip().lower()
            if sentiment == "negative":
                record["mode_2"] = 0
            elif sentiment == "positive":
                record["mode_2"] = 2

        # --- Now use the updated mode_2 ---
        mode_2 = record.get("mode_2")

        if category and category != "Unclear":
            grouped[model_id][category].append(mode_2)


    # Step 2: convert to desired list of dicts
    summary = [{"model_id": model, **cats} for model, cats in grouped.items()]

    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_fuzzy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def fuzzy_aggregation(L, H):
        """Compute fuzzy score using only low (0) and high (2) values."""
        total = L + H
        if total == 0:
            return 0.0
        return H / total  # since weights are 0 for L, 1 for H

    results = []
    for entry in summary:
        model_id = entry["model_id"]
        new_entry = {"model_id": model_id}

        for attr, values in entry.items():
            if attr == "model_id":
                continue

            H = sum(1 for v in values if v == 2)
            L = sum(1 for v in values if v == 0)
            N = len(values)                   # total values incl. neutrals
            LH = L + H                        # only low/high values

            # Decide eligibility for fuzzy scoring
            eligible = (N >= min_count) if count_neutral_toward_min else (LH >= min_count)

            if eligible and LH > 0:
                score = H / LH
            else:
                score = None  # not enough data (or no L/H to score)

            new_entry[attr] = {
                "fuzzy_score": None if score is None else round(score, 3),
                "num_values": N,
                "num_LH": LH,
                "eligible": eligible
            }

        results.append(new_entry)

    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_fuzzy_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

unite_outputs()
exract_output()
fuzzy_aggregation_analysis()

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_fuzzy_full.json") as f:
        a50_all_results = json.load(f)

# Collect all attributes
all_attrs = sorted({k for rec in a50_all_results for k in rec if k != "model_id"})
model_ids = [rec["model_id"] for rec in a50_all_results]

# Create DataFrames
fuzzy_df = pd.DataFrame(index=model_ids, columns=all_attrs, dtype=float)
count_df = pd.DataFrame(index=model_ids, columns=all_attrs, dtype=float)

for rec in a50_all_results:
    mid = rec["model_id"]
    for attr in all_attrs:
        if attr in rec and isinstance(rec[attr], dict):
            fuzzy_df.loc[mid, attr] = rec[attr]["fuzzy_score"]
            count_df.loc[mid, attr] = rec[attr]["num_values"]

# # --- PLOTTING ---

# # Larger figure to make boxes taller
# fig, ax = plt.subplots(figsize=(10, 18))  # increase height here

# data = fuzzy_df.to_numpy(dtype=float)

# # Use a readable diverging colormap
# im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)  # 'auto' lets cells fill space

# # Label axes
# ax.set_xticks(np.arange(len(all_attrs)))
# ax.set_yticks(np.arange(len(model_ids)))
# ax.set_xticklabels(all_attrs, rotation=45, ha="right", fontsize=10)
# ax.set_yticklabels(model_ids, fontsize=9)

# # Colorbar
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2%", pad=0.05)  # 2% of main plot width
# cbar = plt.colorbar(im, cax=cax)
# cbar.set_label("Fuzzy Score", rotation=270, labelpad=15)

# # --- Annotate each cell ---
# # Automatically switch text color based on brightness
# for i in range(len(model_ids)):
#     for j in range(len(all_attrs)):
#         score = fuzzy_df.iat[i, j]
#         nvals = count_df.iat[i, j]
#         if pd.isna(score):
#             continue
#         text = f"{score:.2f}\n(n={int(nvals)})"

#         # Pick contrasting text color
#         color_val = im.cmap(score / np.nanmax(data))
#         brightness = (0.299 * color_val[0] + 0.587 * color_val[1] + 0.114 * color_val[2])
#         text_color = "black" if brightness > 0.5 else "white"

#         ax.text(j, i, text, ha="center", va="center", fontsize=9, color=text_color)

# ax.set_title("Model Quality Attribute Heatmap", pad=20)
# plt.tight_layout()
# # plt.savefig("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_heatmap_A50_min3.png", dpi=300)
# plt.show()