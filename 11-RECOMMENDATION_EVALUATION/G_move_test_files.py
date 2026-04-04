import os
import shutil
import json
# THIS IS FOR EASILT ACCESSING THE TEST MODELS INFORMATION
# NEED TO RUN AGAIN ONCE OUTPUT IS COMPLETED

# Example dictionary (replace with your actual one)
with open("11-RECOMMENDATION_EVALUATION/OUTPUT_F.json", "r") as f:
    data = json.load(f)

SOURCE_DIR = "HF-Models-T7-U"     # folder containing all model json files
DEST_DIR = "8-CRITERIA_SELECTION/test_models"          # destination folder

os.makedirs(DEST_DIR, exist_ok=True)

def model_name_to_filename(model_name: str) -> str:
    """
    Convert 'author/model-name' -> 'author__model-name.json'
    """
    return model_name.strip().replace("/", "__") + ".json"

copied = []
missing = []

for paper in data.values():
    if paper is not None:
        model_field = paper.get("model_full_name", "")
        
        # Split multiple models separated by ';'
        model_names = model_field.split(";")
        
        for model_name in model_names:
            filename = model_name_to_filename(model_name)
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(DEST_DIR, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied.append(filename)
            else:
                missing.append(filename)

print(f"Copied {len(copied)} files:")
for f in copied:
    print("  ", f)

if missing:
    print("\nMissing files:")
    for f in missing:
        print("  ", f)
