import os
import json
import time
import random
import re
from collections import defaultdict
from placeholder_terms import PLACEHOLDER_PATTERNS
from tqdm import tqdm

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

latest_models = []
def has_downloads(data):
    downloads = data.get("likes", 0)
    last_modified = data.get("lastModified", None)

    if last_modified:
        try:
            last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
            # Automatically pass if modified after 2024
            if last_modified_dt.year > 2025:
                latest_models.append(data.get("modelId", "unknown"))
                return False  # Do NOT filter out
        except Exception:
            pass  # ignore bad format
    # Filter only models with zero likes for old models
    return downloads < 1


def has_duplicate(data):
    model_id = data.get("modelId", "")
    model_id1 = model_id.replace("/", "__", 1)
    model_id2 = model_id1 + ".json"
    return model_id2 in duplicate


print(f"Checking directory: {directory}")

# Get all .json files in the directory
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]

# Sample 1% of files (at least 1 file)
SAMPLE = 0.001 # 0.01
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files (seed={RANDOM_SEED}).\n")

filter_counts = {
    "Duplicate": 0,
    "Description": 0,
    "Files": 0,
    "Pipetag": 0,
    "Popularity": 0,
    "Passed": 0,
}

# Loop through sampled files
for filename in tqdm(sampled_files):
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            problems = 0

            if has_description(data):
                # print(f"File {filename} has a placeholder description.")
                problems += 1
                filter_counts["Description"] += 1

            if has_files(data):
                # print(f"File {filename} has only one file.")
                problems += 1
                filter_counts["Files"] += 1

            if has_pipetag(data):
                # print(f"File {filename} has no pipeline tag.")
                problems += 1
                filter_counts["Pipetag"] += 1

            if has_downloads(data):
                # print(f"File {filename} has no downloads or is not modified after 2025.")
                problems += 1
                filter_counts["Popularity"] += 1
            
            if has_duplicate(data):
                problems += 1
                filter_counts["Duplicate"] += 1
            
            if problems == 0:
                # print(f"File {filename} passed all checks.")
                filter_counts["Passed"] += 1
                author = data.get("author", "unknown")
                author_to_files[author].append(filename)
                selected_files.append(filename)

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filename}: {e}")

#OUTPUTS
print("\nSelected Files: \n")
print(selected_files[:10]) # Print first 10 selected files
print(f"\nCount of selected files: {len(selected_files)} \nPercentage: {len(selected_files) / len(sampled_files) * 100:.2f}% \n")

print("Filter Counts:")
for key, count in filter_counts.items():
    print(f"{key}: {count}")


# Get all keys except the last one
import matplotlib.pyplot as plt

total_count = len(sampled_files) 

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
# plt.savefig('N_model_filter.png')
plt.show()


# # MOVE SELECTED FILES
# import shutil

# # Destination folder (you can change this path)
# destination_dir = "HF-Models-PD"
# # Create the destination directory if it doesn't exist
# os.makedirs(destination_dir, exist_ok=True)
# moved = 0
# # Move selected files
# for filename in selected_files:
#     source_path = os.path.join(directory, filename)
#     destination_path = os.path.join(destination_dir, filename)
    
#     try:
#         shutil.copy(source_path, destination_path)
#         # print(f"Moved {filename} to {destination_dir}")
#         moved +=1
#     except Exception as e:
#         print(f"Failed to move {filename}: {e}")

# print(f"\nTotal files moved: {moved} out of {len(selected_files)} selected files.\n")
# print(f"Time taken: {time.time() - start_time:.2f} seconds\n")

