import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random

# ----------- CONFIG -----------
# FOLDER_PATH = "/Users/ardacanseradali/Documents/Thesis_master/HF-Models-v2" # <- Update this
FOLDER_PATH = "/Users/ardacanseradali/Documents/Thesis_master/HF-Models-Y3"  # <- Update this

# ------------------------------
# Get all .json files in the directory
all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# Sample 1% of files (at least 1 file)
SAMPLE = 1 # 0.01
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)


year_set = set()
modified_list = {}

# ----------- DATA EXTRACTION -----------
for filename in tqdm(sampled_files):
    if filename.endswith(".json"):
        file_path = os.path.join(FOLDER_PATH, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                


                # last_modified
                l_modified = data.get("lastModified", None)
                try:
                    last_modified_dt = datetime.fromisoformat(l_modified.replace("Z", "+00:00"))
                    year = last_modified_dt.year
                    if year not in year_set:
                        modified_list[year] = 1
                        year_set.add(year)
                    else: 
                        modified_list[year] += 1

                except:
                    pass

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

with open("1-MODEL_FILTERING/N_Y3_last_modified.json", "w", encoding="utf-8") as f: # WILL BE REMOVED MODELS 
    json.dump(modified_list, f, indent=2, ensure_ascii=False)
