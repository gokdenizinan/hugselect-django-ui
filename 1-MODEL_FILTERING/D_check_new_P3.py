# import B_amodel_filtering
import os
import json
import time
import random
from collections import defaultdict
import spacy
from tqdm import tqdm


# Start the timer
start_time = time.time()
# Set your directory path
directory = "HF-Models-P9"  # Adjust this path as needed
RANDOM_SEED = 45
random.seed(RANDOM_SEED)
# Load and sample files
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
SAMPLE = 1  # Use full dataset
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files.\n")

# Final result dictionary
model_dict = defaultdict(list)

# Main loop
model_count = 0
for filename in tqdm(sampled_files):
    model_count += 1
    file_path = os.path.join(directory, filename)
    key = f"M_{model_count}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_id = data.get("modelId", filename.replace(".json", ""))

            if "/" in model_id:
                author, name = model_id.split("/", 1)
            else:
                author, name = "Unknown", model_id

            model_task = data.get("pipeline_tag", "unknown")

            model_dict[key] = {
                "model_id": model_id,
                "model_name": name,
                "author": data.get("author", author),
                "model_task": model_task,
            }

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Report
elapsed = time.time() - start_time
print(f"\n✅ Extraction completed in {elapsed:.2f} seconds.")
print(f"Processed {sample_size} models.")
print(f"Extracted {len(model_dict)} model entries.")

# Step 1: Remove duplicates by model_name (keep first occurrence)
seen = set()
deduped_dict = {}
for key, entry in model_dict.items():
    name = entry["model_name"]
    if name not in seen:
        deduped_dict[key] = entry
        seen.add(name)

print("------")
print(f"Original: {len(model_dict)} entries")
print(f"After deduplication: {len(deduped_dict)} entries")

# Save results
# os.makedirs("5-REVIEW_COLLECTION", exist_ok=True)
with open("1-MODEL_FILTERING/N_model_dict.json", "w", encoding="utf-8") as out_file:
    json.dump(model_dict, out_file, indent=2)

print("Results saved to model_dict.json.")

with open("5-REVIEW_COLLECTION/model_dict_original.json", "r", encoding="utf-8") as f:
    original_dict = json.load(f)


# --- Extract model_ids ---
ids1 = set(d["model_id"] for d in model_dict.values())
ids2 = set(d["model_id"] for d in original_dict.values())

# --- Sizes ---
size1 = len(model_dict)
size2 = len(original_dict)

# --- Common model_ids ---
common_ids = ids1 & ids2  # intersection
num_common = len(common_ids)

# --- Unique model_ids ---
unique_to_list1 = ids1 - ids2
unique_to_list2 = ids2 - ids1

# --- Print summary ---
print(f"model_dict 1 size: {size1}")
print(f"original_dict 2 size: {size2}")
print(f"Common model_ids: {num_common}")
print(f"Unique to model_dict 1: {len(unique_to_list1)}")
print(f"Unique to original_dict 2: {len(unique_to_list2)}")
# print("Common model_ids:", common_ids)


# --- Unique dictionaries in list2 ---
unique_list2_dicts = [d for d in original_dict.values() if d["model_id"] not in ids1]
unique_list1_dicts = [d for d in model_dict.values() if d["model_id"] not in ids2]


# --- Optional: print or save summary ---
print(f"Number of unique dictionaries in list2: {len(unique_list2_dicts)}")
print(f"Number of unique dictionaries in list1: {len(unique_list1_dicts)}")

# print("Unique dictionaries in list2:")
# for d in unique_list2_dicts:
#     print(d)

# --- Example: save to JSON ---
with open("5-REVIEW_COLLECTION/N_model_dict_run.json", "w", encoding="utf-8") as f:
    json.dump(unique_list1_dicts, f, indent=2, ensure_ascii=False)

with open("5-REVIEW_COLLECTION/N_model_dict_remove.json", "w", encoding="utf-8") as f: # WILL BE REMOVED MODELS 
    json.dump(unique_list2_dicts, f, indent=2, ensure_ascii=False)

# Step 1: Remove duplicates by model_name (keep first occurrence)
seen = set()
deduped_dict = {}
for m_del in unique_list1_dicts:
    m_id = m_del.get("model_id", "")
    na_me =  m_id.split("/")
    name = na_me[1]
    if name not in seen:
        deduped_dict[m_id] = m_del
        seen.add(name)

print("------")
print(f"Original N_model_dict_run: {len(unique_list1_dicts)} entries")
print(f"After deduplication: {len(deduped_dict)} entries")

with open("5-REVIEW_COLLECTION/N_model_dict_dedup_run.json", "w", encoding="utf-8") as f:
    json.dump(deduped_dict, f, indent=2, ensure_ascii=False)