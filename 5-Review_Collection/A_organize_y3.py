import os
import json
import time
import random
from collections import defaultdict
import spacy
from tqdm import tqdm

# Example: list of topics that are already done
with open("5-REVIEW_COLLECTION/done_topics.json", "r", encoding="utf-8") as f:
    done_topics = json.load(f)  # should be a list of topic names


# Start the timer
start_time = time.time()
# Set your directory path
directory = "HF-Models-Y3"  # Adjust this path as needed
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

# # Step 2: Remove entries where model_name is in done_topics
# filtered_dict = {
#     key: entry for key, entry in deduped_dict.items()
#     if entry["model_name"] not in done_topics
# }

print("------")
print(f"Original: {len(model_dict)} entries")
print(f"After deduplication: {len(deduped_dict)} entries")

# # Save results
# os.makedirs("5-REVIEW_COLLECTION", exist_ok=True)
# with open("5-REVIEW_COLLECTION/model_dict.json", "w", encoding="utf-8") as out_file:
#     json.dump(filtered_dict, out_file, indent=2)

print("Results saved to model_dict.json.")
print(len(done_topics))

with open("5-REVIEW_COLLECTION/model_dict_original.json", "w", encoding="utf-8") as out_file:
    json.dump(model_dict, out_file, indent=2)