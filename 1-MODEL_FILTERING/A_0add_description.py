import os
import json

# Paths to folders
small_folder = "1-MODEL_FILTERING/missing_descriptions"
big_folder = "Hf-Models-Y1"

# Loop over files in the small folder
for filename in os.listdir(small_folder):
    if not filename.endswith(".json"):
        continue  # skip non-json files

    small_path = os.path.join(small_folder, filename)
    big_path = os.path.join(big_folder, filename)

    # Only proceed if the same filename exists in big_folder
    if os.path.exists(big_path):
        # Load both JSON files
        with open(small_path, "r", encoding="utf-8") as f:
            small_data = json.load(f)

        with open(big_path, "r", encoding="utf-8") as f:
            big_data = json.load(f)

        # Replace the 'description' key in the big file
        if "description" in small_data:
            big_data["description"] = small_data["description"]

        # Save the updated big JSON back to file
        with open(big_path, "w", encoding="utf-8") as f:
            json.dump(big_data, f, ensure_ascii=False, indent=2)


