import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import pandas as pd
from datetime import datetime


# -------- CONFIGURATION --------
FOLDER_PATH = "HF-Models-Y1"
BATCH_SIZE = 10_000
TIME_WINDOW_DAYS = 1
# --------------------------------

SAMPLE = 1 # 0.01
RANDOM_SEED = 42
random.seed(RANDOM_SEED)



def get_model_fingerprint(row):
    key = (
        str(row.get('author', '')) +
        str(row.get('pipeline_tag', '')) +
        str(row.get('library_name', '')) +
        str(row.get('summary', '')) +
        str(row.get('license', '')) +
        str(row.get('datasets', '')) +
        str(row.get('region', '')) +
        str(row.get('metrics', '')) +
        str(row.get('file_count', '')) +
        str(','.join(sorted(row.get('tags') or [])))
    )
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def parse_date(date_str):
    """Parse date safely, fallback to datetime.min."""
    if pd.isna(date_str):
        return datetime.min
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return datetime.min

def text_similarity(a, b):
    """Compute similarity ratio between two text strings (0–1)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return 0
    return SequenceMatcher(None, a, b).ratio()

def select_best_model(group, similarity_threshold=0.9):
    group = group.copy()
    group['likes'] = pd.to_numeric(group['likes'], errors='coerce').fillna(0)
    group['lastModified_dt'] = group['lastModified'].apply(parse_date)
    group['description'] = group.get('description', '').fillna('')

    # --- Step 1: Group by similarity ---
    groups = []  # each sublist = models that are ≥ similarity_threshold similar
    visited = set()

    for i, row_i in group.iterrows():
        if i in visited:
            continue
        current_group = [i]
        visited.add(i)
        desc_i = row_i['description']

        for j, row_j in group.iterrows():
            if j in visited:
                continue
            sim = text_similarity(desc_i, row_j['description'])
            if sim >= similarity_threshold:
                current_group.append(j)
                visited.add(j)

        groups.append(current_group)

    # --- Step 2: Within each similar group, pick the best model ---
    selected_models = []
    for g in groups:
        subgroup = group.loc[g]
        best = subgroup.sort_values(
            by=['likes', 'lastModified_dt'],
            ascending=[False, False]
        ).iloc[0]
        selected_models.append(best)

    # --- Step 3: Return as DataFrame (only distinct best models) ---
    result_df = pd.DataFrame(selected_models).reset_index(drop=True)
    return result_df


# 📂 Collect all JSON filenames
all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]

sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)
print(f"📁 Total JSON files: {len(sampled_files)}")

selected_models = []
total_processed = 0

# 🔁 Batch processing
for i in range(0, len(sampled_files), BATCH_SIZE):
    batch_files = sampled_files[i:i + BATCH_SIZE]
    print(f"\n📦 Processing batch {i // BATCH_SIZE + 1} with {len(batch_files)} files...")

    records = []
    for filename in tqdm(batch_files):
        path = os.path.join(FOLDER_PATH, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['__filename__'] = filename
                records.append(data)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    if not records:
        continue

    df = pd.DataFrame(records)
    # Create a dictionary mapping fingerprint to model names (or model IDs)
    df.fillna('', inplace=True)
    df['fingerprint'] = df.apply(get_model_fingerprint, axis=1)
    fingerprint_to_models = df.groupby('fingerprint')['modelId'].apply(list).to_dict()
    # Count models
    total_processed += len(df)

    # Detect duplicates
    duplicates = df[df.duplicated('fingerprint', keep=False)]
    uniques = df[~df.duplicated('fingerprint', keep=False)]

    # ✅ Select best from each duplicate group
    selected_from_duplicates = []
    grouped = duplicates.groupby('fingerprint')
    for _, group in grouped:
        best_models_df = select_best_model(group)
        if best_models_df.empty:
            continue
        # If multiple (distinct) models remain after similarity filtering
        if len(best_models_df) > 1:
            for _, row in best_models_df.iterrows():
                selected_from_duplicates.append(row.to_dict())
        else:
            # Only one best model
            selected_from_duplicates.append(best_models_df.iloc[0].to_dict())

    # 📥 Add to final list
    selected_models.extend(selected_from_duplicates)
    selected_models.extend(uniques.to_dict(orient='records'))

# 🎯 Final stats
num_selected = len(selected_models)
percent_selected = (num_selected / total_processed) * 100 if total_processed else 0

print(f"\n✅ Selected {num_selected} models out of {total_processed} total ({percent_selected:.2f}%)")

for model in selected_models:
    model.pop("lastModified_dt", None)

# Optional: Save the selected models to JSON
with open("selected_models.json", "w", encoding='utf-8') as f:
    json.dump(selected_models, f, indent=2)

# Return final percent
print(f"\n📊 Final retained percentage: {percent_selected:.2f}%")


# Get the set of selected filenames
selected_filenames = set(model['__filename__'] for model in selected_models)

# Get the set of selected filenames
selected_filenames = set(model['__filename__'] for model in selected_models)

# Calculate stats
selected_count = len(selected_filenames)
total_files = len(all_files)
selected_percent = (selected_count / total_files) * 100 if total_files else 0

# Print stats
print(f"📄 Number of selected files: {selected_count} out of {total_files} "
      f"({selected_percent:.2f}%)")


# 📊 Plot Selected vs Unselected
selected_count = len(selected_filenames)
unselected_count = sample_size - selected_count

plt.figure(figsize=(6, 6))
bars = plt.bar(["Unique", "Duplicate"], [selected_count, unselected_count], color=["seagreen", "salmon"])
plt.ylabel("Number of Models")
plt.title("Duplicate Model Filtering")
for i, bar in enumerate(bars):
    height = bar.get_height()
    color = bar.get_facecolor()  # Get the bar color
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(int(height)), ha="center", va="bottom", fontsize=12, color=color)
    
plt.tight_layout()
plt.savefig("N_duplicate_filtering.png")
plt.show()

# ---- DUPLICATE ANALYSIS ---- PLOT
import matplotlib.pyplot as plt
from collections import Counter

# Count how many duplicates each fingerprint had
duplicate_counts = Counter(duplicates['fingerprint'])

# Find models with most and least duplicates (excluding unique ones)
if duplicate_counts:
    most_duplicates = duplicate_counts.most_common(5)  # top 5
    least_duplicates = sorted(
        [item for item in duplicate_counts.items() if item[1] > 1],
        key=lambda x: x[1]
    )[:5]  # bottom 5 duplicates

    print("\n📈 Models with most duplicates:")
    for fp, count in most_duplicates:
        print(f"Fingerprint: {fp}, Count: {count}")

    print("\n📉 Models with least duplicates (but >1):")
    for fp, count in least_duplicates:
        print(f"Fingerprint: {fp}, Count: {count}")

# PLOT DUPLICATE COUNTS
# Helper to get display name (truncate to first 10 chars)
def get_display_name(fp):
    models = fingerprint_to_models.get(fp, [])
    if models:
        # Take first model name, truncate to 10 chars
        return models[0][:10] + "..." if len(models[0]) > 12 else models[0]
    return fp[:10] + "..."

# Prepare data for most duplicates
most_duplicates_names = [get_display_name(fp) for fp, _ in most_duplicates]
most_duplicates_counts = [count for _, count in most_duplicates]

# Prepare data for least duplicates (>1)
least_duplicates_names = [get_display_name(fp) for fp, _ in least_duplicates]
least_duplicates_counts = [count for _, count in least_duplicates]

import matplotlib.pyplot as plt

# Plot for Models with Most Duplicates (vertical bars)
plt.figure(figsize=(10, 6))
plt.bar(most_duplicates_names, most_duplicates_counts, color="coral")
plt.ylabel("Number of Duplicates")
plt.title("Models with Most Duplicates")
plt.xticks(rotation=45, ha="right")  # Rotate x labels for readability
plt.tight_layout()
plt.savefig("N_most_duplicates.png")
plt.show()

# Plot for Models with Least Duplicates (>1) (vertical bars)
plt.figure(figsize=(10, 6))
plt.bar(least_duplicates_names, least_duplicates_counts, color="olivedrab")
plt.ylabel("Number of Duplicates")
plt.title("Models with Least Duplicates")
plt.xticks(rotation=45, ha="right")  # Rotate x labels for readability
plt.tight_layout()
plt.savefig("N_least_duplicates.png")
plt.show()

import shutil

### NAMES OF UNSELECTED FILES
import json

# Identify unselected files from the sampled ones
unselected_files = [fname for fname in sampled_files if fname not in selected_filenames]

# Save unselected filenames to JSON
with open("1-MODEL_FILTERING/N_duplicate_model_names.json", "w", encoding="utf-8") as f:
    json.dump(unselected_files, f, indent=2)

print(f"\n📝 Saved {len(unselected_files)} unselected model filenames to 'unselected_models.json'")


### MOVE FILES TO NEW DIRECTORY
NEW_DIR = "HF-Models-P2"
os.makedirs(NEW_DIR, exist_ok=True)

moved_count = 0
for filename in selected_filenames:
    src_path = os.path.join(FOLDER_PATH, filename)
    dst_path = os.path.join(NEW_DIR, filename)
    try:
        shutil.copy(src_path, dst_path)
        moved_count += 1
    except Exception as e:
        print(f"❌ Failed to move {filename}: {e}")

print(f"\n📂 Moved {moved_count} selected files to '{NEW_DIR}' directory.")

