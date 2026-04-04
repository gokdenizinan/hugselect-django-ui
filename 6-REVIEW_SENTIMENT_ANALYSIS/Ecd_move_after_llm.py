import os
import json

# CONFIGURATION
folder_a = "5-REVIEW_COLLECTION/united_f3"
folder_b = "5-REVIEW_COLLECTION/united_run_again_mention"

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/yes_name.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)
with open("1-MODEL_FILTERING/N_model_dict.json", "r", encoding="utf-8") as f:
    n_dict = json.load(f)

model_names_to_find = list(model_dict.keys())

model_ids = [
    info["model_id"]
    for info in n_dict.values()
    if info["model_name"] in model_names_to_find
]

# Normalize filenames: assuming filenames are like "01-ai_Yi-1.5-9B.json"
def model_id_to_filename(model_id):
    return model_id.replace("/", "_") + ".json"

# Count files in each folder
files_a = [f for f in os.listdir(folder_a) if f.endswith(".json")]
files_b = [f for f in os.listdir(folder_b) if f.endswith(".json")]

print(f"📂 Folder A ({folder_a}) contains {len(files_a)} files.")
print(f"📂 Folder B ({folder_b}) contains {len(files_b)} files.\n")

for mid in model_ids:
    file_a = os.path.join(folder_a, model_id_to_filename(mid))
    file_b = os.path.join(folder_b, model_id_to_filename(mid))

    if not os.path.exists(file_a):
        print(f"❌ Missing in folder A: {mid}")
        continue

    # Load A
    with open(file_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    if os.path.exists(file_b):
        # Load from B
        with open(file_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
        # Replace only reddit and stack
        data_a["reddit"] = data_b.get("reddit", [])
        data_a["stack"] = data_b.get("stack", [])
        print(f"🔄 Updated reddit + stack for {mid}")
    else:
        # Empty reddit and stack if no replacement data found
        data_a["reddit"] = []
        data_a["stack"] = []
        print(f"🧹 Emptied reddit + stack for {mid} (no file in B)")

    # Save back
    if data_a["reddit"] == [] and data_a["stack"] == [] and data_a["hf"] == []:
        os.remove(file_a)
    else:
        with open(file_a, "w", encoding="utf-8") as f:
            json.dump(data_a, f, indent=2, ensure_ascii=False)

print("\n✅ Done processing all model_ids.")