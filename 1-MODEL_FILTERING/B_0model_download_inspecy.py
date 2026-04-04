# #BU KODUN AMACI DOWNLOAD SAYISI FILTRE SINIRINI BELIRLEMEK
# import json

# with open("1-MODEL_FILTERING/hf_model_download_stats.json", "r", encoding = "utf-8") as f:
#     p4_download = json.load(f)

# # Define your thresholds
# MIN_ALL_TIME = 50
# MIN_LAST_30_DAYS = 3

# # Filter models
# def download_filter(models):
#     model_ids = [
#         name
#         for model in models
#         for name, stats in model.items()
#         if stats["downloads_all_time"] < 50
#         and stats["downloads_last_30_days"] < 3
#     ]
#     return model_ids

# popular_models = download_filter(p4_download)
# for model in popular_models[:10]:
#         print(model)
# print(len(p4_download))
# print(len(popular_models))

# import os
# import json
# import time
# import random
# import re
# from collections import defaultdict
# from datetime import datetime
# from placeholder_terms import PLACEHOLDER_PATTERNS
# from tqdm import tqdm

# ## BU KODUN AMACI DOWNLOAD SAYISI FILTRESINI ENTEGRE ETMEK AYNI ZAMANDA DUPLICATE FILTRESINI GRAFIKLE GOSTERMEK
# # Start the timer
# start_time = time.time()

# # Set your directory path
# # Set a random seed for reproducibility
# RANDOM_SEED = 42
# random.seed(RANDOM_SEED)

# # List to store results
# selected_files = []
# author_to_files = defaultdict(list)

# # Set your directory path
# directory = "HF-Models-Y1"

# print(f"Checking directory: {directory}")

# # Get all .json files in the directory
# all_files = [f for f in os.listdir(directory) if f.endswith(".json")]

# # Sample 1% of files (at least 1 file)
# SAMPLE = 1 # 0.01
# sample_size = max(1, int(len(all_files) * SAMPLE))
# sampled_files = random.sample(all_files, sample_size)

# print(f"Randomly selected {sample_size} of {len(all_files)} files (seed={RANDOM_SEED}).\n")

# model_id_like = {}
# for filename in tqdm(sampled_files):
#     file_path = os.path.join(directory, filename)
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             like = data.get("likes", 0)
#             if like > 0:
#                 m_id = data.get("modelId", "")
#                 model_id_like[m_id] = like
#     except (json.JSONDecodeError, IOError) as e:
#         print(f"Error reading {filename}: {e}")

# # Merge likes into a copy of p4_download
# p4_download_copy = []
# for item in p4_download:
#     model_id = list(item.keys())[0]  # assuming one key per item
#     model_data = item[model_id]      # original stats dict
#     if model_id in model_id_like:
#         model_data = model_data.copy()  # avoid modifying original
#         model_data["likes"] = model_id_like[model_id]
#     p4_download_copy.append({model_id: model_data})


        
# with open("1-MODEL_FILTERING/p4_download_copy.json", "w", encoding="utf-8") as f: # WILL BE REMOVED MODELS 
#     json.dump(p4_download_copy, f, indent=2, ensure_ascii=False)   

import json
with open("1-MODEL_FILTERING/p4_download_copy.json", "r", encoding = "utf-8") as f:
    p4_download = json.load(f)

import pandas as pd

# Flatten your list of dicts into a DataFrame
rows = []
for item in p4_download:
    for model_id, stats in item.items():
        row = stats.copy()
        row["modelId"] = model_id
        rows.append(row)

df = pd.DataFrame(rows)

# Calculate correlation
correlation = df["downloads_last_30_days"].corr(df["likes"])
print(f"Correlation between downloads_last_30_days and likes: {correlation:.3f}")
