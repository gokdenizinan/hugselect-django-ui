import os
import json

# Path to your folder containing JSON files
folder_path = "HF-Models-T7"

# List to store modelID values
model_ids = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                
                # If modelID is at the top level
                if "modelID" in data:
                    model_ids.append(data["modelID"])
                
                # If modelID might be nested (optional)
                # You can expand this logic if needed
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")

# Output file
output_file = "8-CRITERIA_SELECTION/model_ids_list.json"

# Save list to JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(model_ids, f, indent=4)

print(f"Extracted {len(model_ids)} modelIDs and saved to {output_file}")
