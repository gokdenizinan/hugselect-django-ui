import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_fuzzy_full.json", "r", encoding="utf-8") as f:
    data = json.load(f)


rows = []
for d in data:
    model = d["model_id"]
    for attr, vals in d.items():
        if attr == "model_id":
            continue

        if not vals.get("eligible"):   # <-- keep only eligible
            continue

        rows.append({
            "model_id": model,
            "attribute": attr,
            "fuzzy": vals.get("fuzzy_score"),
            "LH": vals.get("num_LH"),
        })

df = pd.DataFrame(rows)

# top 20 models by LH
top_models = (
    df.groupby("model_id")["LH"]
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .index
)

df = df[df["model_id"].isin(top_models)]

fuzzy_matrix = df.pivot_table(index="model_id", columns="attribute", values="fuzzy", aggfunc="first")
lh_matrix = df.pivot_table(index="model_id", columns="attribute", values="LH", aggfunc="first")

df["annot"] = df.apply(
    lambda r: f"{r['fuzzy']:.2f}\nLH:{r['LH']}",
    axis=1
)
annot_matrix = df.pivot_table(index="model_id", columns="attribute", values="annot", aggfunc="first")

plt.figure(figsize=(12,10))
ax = sns.heatmap(fuzzy_matrix, cmap="RdYlGn", vmin=0, vmax=1, annot=annot_matrix, fmt="")
ax.set_xlabel("Quality Attributes")
ax.set_ylabel("Models")
ax.set_title("")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("13-GRAPHS/Quality_Attribute_Heatmap/quality_attribute_heatmap.png", dpi=150)