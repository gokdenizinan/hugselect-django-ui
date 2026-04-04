import os
import json
import time
import random
from collections import defaultdict, Counter
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
import emoji
import re
import datetime
DetectorFactory.seed = 0  # makes detection reproducible



# Start the timer
start_time = time.time()
# Set your directory path
directory = "HF-Models-P9"  # Adjust this path as needed
RANDOM_SEED = 45
random.seed(RANDOM_SEED)

print(f"Checking directory: {directory}")

# Get all .json files in the directory
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
# Sample 1% of files (at least 1 file)
SAMPLE = 1 # 0.01
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files (seed={RANDOM_SEED}).\n")

# Loop through sampled files
model_likes = {}
for filename in sampled_files:
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_name = data.get("modelId", 0)
            like_count = data.get("likes", 0)
            model_likes[model_name] = like_count
            
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

sorted_model_likes = dict(sorted(model_likes.items(), key=lambda item: item[1], reverse=True))

file_path = f"1-MODEL_FILTERING/N_sorted_model_likes_P9.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(sorted_model_likes, f, indent=2, ensure_ascii=False)

first_10k = dict(list(sorted_model_likes.items())[:10000])

file_path = f"1-MODEL_FILTERING/N_model_likes_10k_P9.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(first_10k, f, indent=2, ensure_ascii=False)