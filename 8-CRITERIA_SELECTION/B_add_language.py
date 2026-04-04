
import os
import json
import csv

MODELS_FOLDER = "HF-Models-T7"
ISO_TAB_PATH = "8-CRITERIA_SELECTION/iso-639-3.tab"

# --- Build language mapping from ISO file ---
code_to_name = {}

with open(ISO_TAB_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:
        iso3 = row["Id"].strip().lower()        # 3-letter
        iso1 = row["Part1"].strip().lower()     # 2-letter (may be empty)
        name = row["Ref_Name"].strip()

        if iso3:
            code_to_name[iso3] = name
        if iso1:
            code_to_name[iso1] = name

print(f"Loaded {len(code_to_name)} language codes.")

# --- Update model files ---
updated = 0

for filename in os.listdir(MODELS_FOLDER):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(MODELS_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    metadata = model.get("Metadata", {})
    langs = metadata.get("language")

    if isinstance(langs, list):
        new_langs = []
        for code in langs:
            if isinstance(code, str):
                full = code_to_name.get(code.lower(), code)
                new_langs.append(full)
            else:
                new_langs.append(code)

        metadata["language"] = new_langs
        model["Metadata"] = metadata

        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

        updated += 1

print(f"Updated {updated} files.")
