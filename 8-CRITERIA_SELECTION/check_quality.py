import os
import json

folder_path = "HF-Models-T8"

count = 0

for filename in os.listdir(folder_path):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(folder_path, filename)

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # skip if "quality" is missing or not a dict
        quality = data.get("Quality")
        if not isinstance(quality, dict):
            continue

        # check if ANY key in "quality" has a value assigned
        # (i.e., not None, not empty string, not empty list/dict)
        if any(v for v in quality.values()):
            count += 1

    except Exception as e:
        print(f"Error reading {filename}: {e}")

print("Number of files with at least one 'quality' value:", count)