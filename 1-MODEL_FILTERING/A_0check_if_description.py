import os
import json
import time
import random
import re
from collections import defaultdict
from placeholder_terms import PLACEHOLDER_PATTERNS
from datetime import datetime


# Start the timer
start_time = time.time()

# Set your directory path
directory = "HF-Models-Y1"


def has_description(data):
    description = data.get("description", None)
    if not description or len(description.strip()) < 10:
        return True
    return False

def has_files(data):
    files = data.get("file_count", None)
    return files <=1

def has_pipetag(data):
    likes = data.get("likes", 0)
    
    tag = data.get("pipeline_tag", None)
    if likes < 20:
        return tag == [] or tag is None
    else:
        return False

def has_downloads(data):
    downloads = data.get("likes", 0)
    last_modified = data.get("lastModified", None)

    if last_modified:
        try:
            last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            # Automatically pass if modified after 2024
            if last_modified_dt.year > 2025:
                return False  # Do NOT filter out
        except Exception:
            pass  # ignore bad format
    # Filter only models with zero likes for old models
    return downloads < 1



# Step 1: Collect likes from all files
model_likes = {}
file_map = {}  # map modelId -> filename

for filename in os.listdir(directory):
    if not filename.endswith(".json"):
        continue
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_id = data.get("modelId", None)
            likes = data.get("likes", 0)
            if model_id:
                model_likes[model_id] = likes
                file_map[model_id] = filename
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

# Step 2: Sort by likes and take top 10k
sorted_model_likes = dict(sorted(model_likes.items(), key=lambda item: item[1], reverse=True))
top_all_models = dict(list(sorted_model_likes.items()))


no_desc = []
# Step 3: Filter only top 10k files
author_to_files = defaultdict(list)


for model_id in top_all_models.keys():
    filename = file_map[model_id]
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if has_files(data) or has_downloads(data) or has_pipetag(data):
                continue
            if has_description(data):
                no_desc.append(model_id)

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

print("done")
# Save to JSON for reference
with open("1-MODEL_FILTERING/no_desc_2.json", "w", encoding="utf-8") as f:
    json.dump(no_desc, f, indent=2, ensure_ascii=False)