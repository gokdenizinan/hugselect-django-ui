import os
import json
import time
import random
import re
from collections import defaultdict
from datetime import datetime
from placeholder_terms import PLACEHOLDER_PATTERNS
from tqdm import tqdm

## BU KODUN AMACI DOWNLOAD SAYISI FILTRESINI ENTEGRE ETMEK AYNI ZAMANDA DUPLICATE FILTRESINI GRAFIKLE GOSTERMEK
# Start the timer
start_time = time.time()

# Set your directory path
# Set a random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# List to store results
selected_files = []
author_to_files = defaultdict(list)

# Set your directory path
directory = "HF-Models-Y1"


with open("1-MODEL_FILTERING/N_duplicate_model_names.json", "r", encoding = "utf-8") as f:
    duplicate = json.load(f)

with open("1-MODEL_FILTERING/hf_model_download_stats.json", "r", encoding = "utf-8") as f:
    p4_download = json.load(f)

# with open("1-MODEL_FILTERING/hf_model_download_stats_2024.json", "r", encoding = "utf-8") as f:
#     p4_download_2024 = json.load(f)

print(f"Checking directory: {directory}")

# Get all .json files in the directory
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]

# Sample 1% of files (at least 1 file)
SAMPLE = 0.01 # 0.01
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files (seed={RANDOM_SEED}).\n")

filter_counts = {
    "Duplicate": 0,
    "Description": 0,
    "Files": 0,
    "Tags": 0,
    "Popularity": 0,
    "Passed": 0,
}

top_10_fail = defaultdict()
last_error = []
# Loop through sampled files
for filename in tqdm(sampled_files):
    file_path = os.path.join(directory, filename)
    p_count = {
        "Duplicate": 0,
        "Description": 0,
        "Files": 0,
        "Tags": 0,
        "Popularity": 0,
    }
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            last_modified = data.get("lastModified", None)
            likes =  data.get("likes", 0)

            try:
                last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
                last_error.append(last_modified_dt.year)
            except Exception:
                last_error.append(last_modified)
                pass

            
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

print(len(last_error))

with open("1-MODEL_FILTERING/last_error.json", "w", encoding="utf-8") as f: # WILL BE REMOVED MODELS 
    json.dump(last_error, f, indent=2, ensure_ascii=False)