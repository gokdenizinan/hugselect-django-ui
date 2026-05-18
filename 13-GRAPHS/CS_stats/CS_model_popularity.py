import json
from pathlib import Path

with open('8-CRITERIA_SELECTION/F_Hits_United/combined_recommendations.json', 'r') as f:
    combined = json.load(f)

DICT_FOLDER = Path("HF-Models-T7-U")
# DICT_FOLDER = Path("Thesis_master/HF-Models-v2")



from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
RECS_CSV = Path("8-CRITERIA_SELECTION/hits_cluster/experiment_runs_XX_dedup/multisource/ground_truth_vs_recommendations_grouped.csv")
DICT_FOLDER = Path("HF-Models-T7-U")
OUTPUT_DIR = Path("13-GRAPHS/CS_stats")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEMS = ["recsys", "chatgpt", "gemini", "claude", "perplexity"]


# ----------------------------
# Load HF knowledgebase
# ----------------------------
def load_hf_model_database(dict_folder: Path):
    """
    Loads all JSON files in DICT_FOLDER and returns:
    - model_lookup: dict mapping modelID -> full metadata dict
    """

    model_lookup = {}

    json_files = list(dict_folder.rglob("*.json"))

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Case 1: file contains one model dict
            if isinstance(data, dict):
                model_id = data.get("modelID")
                if model_id:
                    model_lookup[model_id] = data

            # Case 2: file contains a list of model dicts
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("modelID")
                        if model_id:
                            model_lookup[model_id] = item

        except Exception as e:
            print(f"Could not read {file}: {e}")

    return model_lookup


model_db = load_hf_model_database(DICT_FOLDER)
print(f"Loaded {len(model_db):,} models from knowledgebase.")


# ----------------------------
# Load recommendations CSV
# ----------------------------
# ----------------------------
# Load recommendations CSV with 2-row header
# ----------------------------
df = pd.read_csv(RECS_CSV, header=[0, 1])

# Clean column names
df.columns = pd.MultiIndex.from_tuples([
    (
        str(level0).strip().lower(),
        str(level1).strip().lower()
    )
    for level0, level1 in df.columns
])

SYSTEMS = ["recsys", "chatgpt", "gemini", "claude", "perplexity"]

# Helpful debug print
print("\nAvailable grouped columns:")
for col in df.columns:
    print(col)

# ----------------------------
# Identify metadata columns
# ----------------------------
sample_col = ("metadata", "sample")
rank_col = ("metadata", "recommendation_rank")

if sample_col not in df.columns:
    raise ValueError("Could not find ('metadata', 'sample') column.")

if rank_col not in df.columns:
    raise ValueError("Could not find ('metadata', 'recommendation_rank') column.")

# ----------------------------
# Select recommendation model_id columns only
# ----------------------------
model_id_cols = [
    ("model_id", system)
    for system in SYSTEMS
    if ("model_id", system) in df.columns
]

missing_model_id_cols = [
    system for system in SYSTEMS
    if ("model_id", system) not in df.columns
]

if missing_model_id_cols:
    raise ValueError(
        f"Missing model_id recommendation columns for: {missing_model_id_cols}"
    )

# ----------------------------
# Convert wide to long
# ----------------------------
records = []

for system in SYSTEMS:
    col = ("model_id", system)

    temp = df[[sample_col, rank_col, col]].copy()
    temp.columns = ["sample", "recommendation_rank", "recommended_model_id"]
    temp["system"] = system

    records.append(temp)

long_df = pd.concat(records, ignore_index=True)

long_df = long_df.dropna(subset=["recommended_model_id"])
long_df["recommended_model_id"] = (
    long_df["recommended_model_id"]
    .astype(str)
    .str.strip()
)

print(long_df.head())

# ----------------------------
# Check existence in knowledgebase
# ----------------------------
long_df["exists_in_kb"] = long_df["recommended_model_id"].isin(model_db)


# ----------------------------
# Extract downloads_all_time
# ----------------------------
def get_downloads_all_time(model_id):
    if model_id not in model_db:
        return None

    metadata = model_db[model_id].get("Metadata", {})
    return metadata.get("downloads_all_time")


long_df["downloads_all_time"] = long_df["recommended_model_id"].apply(get_downloads_all_time)
long_df["downloads_all_time"] = pd.to_numeric(long_df["downloads_all_time"], errors="coerce")


# ----------------------------
# Save evaluated recommendations
# ----------------------------
long_df.to_csv(OUTPUT_DIR / "recommendation_kb_evaluation_long.csv", index=False)


# ----------------------------
# Plot 1: Missing recommendations by system
# ----------------------------
missing_summary = (
    long_df
    .groupby("system")
    .agg(
        total_recommendations=("recommended_model_id", "count"),
        missing_from_kb=("exists_in_kb", lambda x: (~x).sum())
    )
    .reset_index()
)

missing_summary["missing_percent"] = (
    missing_summary["missing_from_kb"] / missing_summary["total_recommendations"] * 100
)

missing_summary.to_csv(OUTPUT_DIR / "missing_from_kb_summary.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(missing_summary["system"], missing_summary["missing_percent"])
plt.ylabel("Recommendations missing from knowledgebase (%)")
plt.xlabel("System")
plt.title("Missing Model Recommendations by System")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_recommendations_by_system.png", dpi=300)
plt.show()


# ----------------------------
# Plot 2: Violin plot of downloads_all_time
# ----------------------------
plot_df = long_df.dropna(subset=["downloads_all_time"]).copy()

# Optional: log transform because downloads are usually very skewed
plot_df["log10_downloads_all_time"] = (
    plot_df["downloads_all_time"] + 1
).apply(lambda x: __import__("math").log10(x))

import matplotlib.pyplot as plt

# ----------------------------
# Pretty system names
# ----------------------------
pretty_names = {
    "recsys": "HugSelect",
    "chatgpt": "ChatGPT",
    "gemini": "Gemini",
    "claude": "Claude",
    "perplexity": "Perplexity"
}

systems_order = ["recsys", "chatgpt", "gemini", "claude", "perplexity"]

# Prepare data in correct order
data = [
    plot_df.loc[plot_df["system"] == system, "log10_downloads_all_time"]
    for system in systems_order
]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(10, 6))

parts = plt.violinplot(
    data,
    showmeans=True,
    showmedians=True
)

# ----------------------------
# Styling violins
# ----------------------------
for pc in parts['bodies']:
    pc.set_facecolor("#8FBBD9")   # soft blue
    pc.set_edgecolor("#4A6FA5")
    pc.set_alpha(0.8)

for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    vp = parts[partname]
    vp.set_edgecolor("#2F3E46")
    vp.set_linewidth(1.2)

# ----------------------------
# Axis formatting
# ----------------------------
plt.xticks(
    range(1, len(systems_order) + 1),
    [pretty_names[s] for s in systems_order],
    fontsize=11
)

plt.ylabel("Downloads (log10)", fontsize=12)
plt.xlabel("")  # remove x-axis label

# Remove top/right spines for cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Slight grid for readability
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# Save
plt.savefig(OUTPUT_DIR / "downloads_all_time_violin_by_system.png", dpi=300)
plt.show()



# ----------------------------
# Optional: print summary table
# ----------------------------
popularity_summary = (
    plot_df
    .groupby("system")["downloads_all_time"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

print("\nMissing-from-KB summary:")
print(missing_summary)

print("\nDownloads popularity summary:")
print(popularity_summary)


# ----------------------------
# Unique missing model_ids per system → JSON
# ----------------------------
missing_models_per_system = (
    long_df[~long_df["exists_in_kb"]]              # keep only missing
    .groupby("system")["recommended_model_id"]
    .apply(lambda x: sorted(set(x)))               # unique + sorted
    .to_dict()
)

# Optional: ensure all systems appear (even if empty)
for system in SYSTEMS:
    missing_models_per_system.setdefault(system, [])

# Save to JSON
output_path = OUTPUT_DIR / "missing_model_ids_per_system.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(missing_models_per_system, f, indent=2)

print(f"\nSaved missing model IDs per system → {output_path}")