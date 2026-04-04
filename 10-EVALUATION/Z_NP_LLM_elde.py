import json

def extract_valid_dicts(text):
    results = []
    decoder = json.JSONDecoder()
    i = 0

    while i < len(text):
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text[i:])
                results.append(obj)
                i += end
            except json.JSONDecodeError:
                # Incomplete or invalid JSON → skip
                i += 1
        else:
            i += 1

    return results


def extract_valid_output(dict):
    all_results = []
    for key, val in dict.items():
        text = val["output"]
        valid_dicts = extract_valid_dicts(text)
        all_results.extend(valid_dicts)
    return all_results

def count_incomplete_dicts(
    data,
    required_keys=("phrase", "label", "reason"),
    allow_empty=False,
):

    incomplete = 0

    for d in data:
        if not isinstance(d, dict):
            incomplete += 1
            continue

        for key in required_keys:
            if key not in d:
                incomplete += 1
                break

            if not allow_empty and (d[key] is None or d[key] == ""):
                incomplete += 1
                break

    return incomplete

import json

import json

def build_drop_reason_lookup(cleaned_output, save_path):
    """
    Build a noun_phrase -> {drop: yes/no, reason: ...} dictionary,
    sorted by drop value first, then reason, then phrase.
    """

    lookup = {}

    # 1. Build lookup
    for item in cleaned_output:
        phrase = item.get("phrase")
        label = item.get("label")
        reason = item.get("reason", "unknown")

        if phrase is None:
            continue

        lookup[phrase] = {
            "drop": "yes" if label == "drop" else "no",
            "reason": reason
        }

    # 2. Sort: drop -> reason -> phrase
    sorted_items = sorted(
        lookup.items(),
        key=lambda x: (
            0 if x[1]["drop"] == "yes" else 1,  # drop first
            x[1]["reason"],
            x[0]
        )
    )

    sorted_lookup = dict(sorted_items)

    # # 3. Save
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(sorted_lookup, f, indent=4, ensure_ascii=False)

    return sorted_lookup



import json
from pathlib import Path

base_dir = Path("10-EVALUATION/llm_nps")

valid_dicts = []

for path in sorted(base_dir.glob("output_G4*.json")):
    with open(path, "r") as file:
        output = json.load(file)
    valid_output = extract_valid_output(output)
    # llm outputundan kac tane noun phrase label dicti cikmis
    # print("In path:", path)
    # print("Total incomplete dicts:", count_incomplete_dicts(output))
    # print("Valid dicts extracted:", len(valid_output))
    valid_dicts.extend(valid_output)



# with open("10-EVALUATION/llm_nps/cleaned_output_G4.json", "w") as outfile:
#     json.dump(valid_dicts, outfile, indent=4)

# NP_PP MISSING LISTESI
with open("2-NP_EXTRACTION/NP_G5_list.json", "r") as file:
    pp_list = json.load(file)

with open("10-EVALUATION/llm_nps/cleaned_output_G4.json", "r") as file:
    cleaned_output = json.load(file)
    
drop_dict = build_drop_reason_lookup(
    cleaned_output,
    "2-NP_EXTRACTION/NP_G5_dropreason.json"
)

print("Total phrases in drop lookup:", len(drop_dict))

phrases_to_remove = {r["phrase"] for r in cleaned_output if "phrase" in r}
print("number of output nps", len(phrases_to_remove))

# 2. Filter names
cleaned_pp_list = [name for name in pp_list if name not in phrases_to_remove]

# 3. Save to JSON
# with open("2-NP_EXTRACTION/NP_comfy_GG_list_half.json", "w", encoding="utf-8") as f:
#     json.dump(cleaned_pp_list, f, indent=4, ensure_ascii=False)

# with open("2-NP_EXTRACTION/NP_G_checked.json", "w", encoding="utf-8") as f:
#     json.dump(list(phrases_to_remove), f, indent=4, ensure_ascii=False)

print("done")

# DUPLICATE CHECK


from collections import Counter

phrases = [d["phrase"] for d in valid_dicts if "phrase" in d]

counts = Counter(phrases)
duplicates = {p: c for p, c in counts.items() if c > 1}
print("dupilcate outputs for nps",len(duplicates))
for i in duplicates:
    print(i, duplicates[i])

# MOVE TO description to panda 

with open("2-NP_EXTRACTION/NP_G5.json", "r", encoding="utf-8") as f:
    np_info = json.load(f)

import re


label_lookup = {
    item["phrase"]: item["label"]
    for item in cleaned_output
}

new_data = {}
idx = 0

total_lookup_hits = 0
unique_lookup_keys = set()

for _, value in np_info.items():
    normalized_np = value["noun_phrase"]

    label = label_lookup.get(normalized_np)

    # total count (includes duplicates)
    if label is not None:
        total_lookup_hits += 1
        unique_lookup_keys.add(normalized_np)

    # drop ONLY if explicitly labeled "drop"
    if label == "drop":
        continue

    # otherwise keep
    if label is not None:
        value["label"] = label

    new_data[str(idx)] = value
    idx += 1

print("Total lookup hits:", total_lookup_hits)
print("Unique lookup items used:", len(unique_lookup_keys))

# BASKA DUMPLARI DA COMMENT OUTLADIM EGER LAZIM OLAN VARSA UNCOMMENT

# BURADAN DEVA GGG DEN ABBREVIATION SONRA ACCURACY
# with open("2-NP_EXTRACTION/NP_comfy_GGG.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)

#SUAN KI X XX COP
# with open("4-LLM_FEATURE_ORGANIZATION/NP_comfy_XXX.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, indent=4, ensure_ascii=False)

print("original PP:", len(np_info))
print("filtreli PPP:", len(new_data))
