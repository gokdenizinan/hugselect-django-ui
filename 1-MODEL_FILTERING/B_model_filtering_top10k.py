import os
import json
import time
import random
import re
from collections import defaultdict
from placeholder_terms import PLACEHOLDER_PATTERNS

# Start the timer
start_time = time.time()

# Set your directory path
directory = "HF-Models-P2"


def has_description(data):
    description = data.get("description", None)
    if not description or len(description.strip()) < 100: # too short
        return True
    elif len(set(description.split())) < 30: # too short or generic
        if description.count("model") > 5:
            return True
        elif len(set(description.split())) < 20:
            return True    
    
    desc = description.lower().strip()
    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, desc):
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

from datetime import datetime

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
top_10k_models = dict(list(sorted_model_likes.items())[:5000])
top_all_models = list(sorted_model_likes.items())

# Save for reference
with open("1-MODEL_FILTERING/model_likes_5k_Y2.json", "w", encoding="utf-8") as f:
    json.dump(top_10k_models, f, indent=2, ensure_ascii=False)

ne_bunlar_amk = []
# Step 3: Filter only top 10k files
selected_files = []
author_to_files = defaultdict(list)
filter_counts = {
    "Description": 0,
    "Files": 0,
    "Pipetag": 0,
    "Popularity": 0,
    "Passed": 0,
}

for model_id in top_10k_models.keys():
    filename = file_map[model_id]
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problems = 0

            if has_description(data):
                problems += 1
                filter_counts["Description"] += 1
                ne_bunlar_amk.append(model_id)

            if has_files(data):
                problems += 1
                filter_counts["Files"] += 1
                print(f"File {filename} has only one file.")
                print(f"{model_id}")

            if has_pipetag(data):
                problems += 1
                filter_counts["Pipetag"] += 1

            if has_downloads(data):
                problems += 1
                filter_counts["Popularity"] += 1

            if problems == 0:
                filter_counts["Passed"] += 1
                author = data.get("author", "unknown")
                author_to_files[author].append(filename)
                selected_files.append(filename)

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")


#OUTPUTS
print("\nSelected Files: \n")
print(selected_files[:10]) # Print first 10 selected files
print(f"\nCount of selected files: {len(selected_files)} \nPercentage: {len(selected_files) / len(top_10k_models) * 100:.2f}% \n")

print("Filter Counts:")
for key, count in filter_counts.items():
    print(f"{key}: {count}")

# Dictionary to store likes of filtered out models
filtered_out_likes = {}

for model_id in top_10k_models.keys():
    filename = file_map[model_id]
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problems = 0

            if has_description(data):
                problems += 1
            if has_files(data):
                problems += 1
            if has_pipetag(data):
                problems += 1
            if has_downloads(data):
                problems += 1

            # If any filter triggered, add to filtered_out_likes
            if problems > 0:
                filtered_out_likes[model_id] = data.get("likes", 0)

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

with open("1-MODEL_FILTERING/ne_bunlar_amk.json", "w", encoding="utf-8") as f:
    json.dump(ne_bunlar_amk, f, indent=2, ensure_ascii=False)

# Save to JSON for reference
with open("1-MODEL_FILTERING/filtered_out_5k.json", "w", encoding="utf-8") as f:
    json.dump(filtered_out_likes, f, indent=2, ensure_ascii=False)

print(f"Filtered out {len(filtered_out_likes)} models.")


# Get all keys except the last one
import matplotlib.pyplot as plt

total_count = len(top_10k_models) 

# Update counts for failed filters
for key in list(filter_counts.keys())[:-1]:
    filter_counts[key] = total_count - filter_counts[key]

# Prepare data for plotting
labels = list(filter_counts.keys())
counts = list(filter_counts.values())

# Separate failed and passed bars
failed_labels = labels[:-1]
failed_counts = counts[:-1]
passed_label = labels[-1]
passed_count = counts[-1]

plt.figure(figsize=(10,6))

# Plot bars
plt.bar(failed_labels, failed_counts, color='lightseagreen', label='Passed filters')
passed_bar = plt.bar(passed_label, passed_count, color='seagreen', label='Passed all filters')

# Add total line
plt.axhline(y=total_count, color='gray', linestyle='--', label='Total sampled files')

# Add passed count above green bar
plt.text(
    passed_bar[0].get_x() + passed_bar[0].get_width()/2,
    passed_bar[0].get_height() + 1,
    str(passed_count),
    ha='center',
    va='bottom',
    fontsize=10,
    color='green'
)

plt.ylabel('Number of Models')
plt.title('Model Filter Criteria')
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
plt.tight_layout()
plt.savefig('model_filter_10k.png')
plt.show()


# MOVE SELECTED FILES
import shutil

# Destination folder (you can change this path)
destination_dir = "HF-Models-Y3B"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)
moved = 0
# Move selected files
for filename in selected_files:
    source_path = os.path.join(directory, filename)
    destination_path = os.path.join(destination_dir, filename)
    
    try:
        shutil.copy(source_path, destination_path)
        # print(f"Moved {filename} to {destination_dir}")
        moved +=1
    except Exception as e:
        print(f"Failed to move {filename}: {e}")

print(f"\nTotal files moved: {moved} out of {len(selected_files)} selected files.\n")
print(f"Time taken: {time.time() - start_time:.2f} seconds\n")



# Dictionary to store likes of filtered out models
filtered_out_likes = {}

for model_id in top_all_models.keys():
    filename = file_map[model_id]
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problems = 0

            if has_description(data):
                problems += 1
            if has_files(data):
                problems += 1
            if has_pipetag(data):
                problems += 1
            if has_downloads(data):
                problems += 1

            # If any filter triggered, add to filtered_out_likes
            if problems > 0:
                filtered_out_likes[model_id] = data.get("likes", 0)

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

# Save to JSON for reference
with open("1-MODEL_FILTERING/filtered_out_all.json", "w", encoding="utf-8") as f:
    json.dump(filtered_out_likes, f, indent=2, ensure_ascii=False)