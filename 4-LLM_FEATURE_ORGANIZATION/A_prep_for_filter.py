import copy
import json

KEEP_KEYS = {
    "noun_phrase",
    "model_id",
    "sentence",
    "representative_sentences",
    "prompt",
    "output",
    "model_used",
}

with open("4-LLM_FEATURE_ORGANIZATION/NP_info.json", "r", encoding="utf-8") as f:
    data = json.load(f)

result = {}

for outer_id, entry in data.items():
    entry = copy.deepcopy(entry)

    # ---- Clean noun_phrase ----
    if isinstance(entry.get("noun_phrase"), str):
        entry["noun_phrase"] = entry["noun_phrase"].split("(")[0].strip()

    # ---- Extract global_frequency as count ----
    count = None
    try:
        count = int(entry.get("np_metadata", {}).get("global_frequency"))
    except (TypeError, ValueError):
        pass

    # ---- Build cleaned entry ----
    cleaned_entry = {
        k: entry[k]
        for k in KEEP_KEYS
        if k in entry
    }

    cleaned_entry["count"] = count

    result[outer_id] = cleaned_entry

with open("4-LLM_FEATURE_ORGANIZATION/NP_info_go.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False) 