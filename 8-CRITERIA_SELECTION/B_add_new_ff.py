import os
import json
from collections import defaultdict

# --- INPUTS ---
MODELS_FOLDER = "HF-Models-T7"      # folder of per-model json dicts
NP_GG_FIN_PATH = "4-LLM_FEATURE_ORGANIZATION/NP_GG_fin.json"    # source dictionary file

# --- LOAD NP_GG_fin ---
with open(NP_GG_FIN_PATH, "r", encoding="utf-8") as f:
    NP_GG_fin = json.load(f)

# --- BUILD model_id -> phrases index ---
model_to_phrases = defaultdict(set)

for _, entry in NP_GG_fin.items():
    if not isinstance(entry, dict):
        continue

    phrase = entry.get("noun_phrase")
    if not phrase:
        continue

    model_ids = entry.get("model_id") or entry.get("modelID") or entry.get("model_ids")
    if not isinstance(model_ids, list):
        continue

    for mid in model_ids:
        if isinstance(mid, str) and mid.strip():
            model_to_phrases[mid.strip()].add(phrase)

# --- UPDATE EACH MODEL JSON ---
updated = 0
missing = 0

for filename in os.listdir(MODELS_FOLDER):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(MODELS_FOLDER, filename)

    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    model_id = model.get("modelID") or model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        continue
    model_id = model_id.strip()

    phrases = sorted(model_to_phrases.get(model_id, set()))

    # Replace Features list
    model["Features"] = phrases

    if phrases:
        updated += 1
    else:
        missing += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

print(f"Updated Features for {updated} models. No phrases found for {missing} models.")
