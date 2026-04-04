import os, json, random, statistics
from pathlib import Path

random.seed(55) # 42 eskiden

FOLDER = Path("HF-Models-T6")   # change me
N_PER_GROUP = 250
OUTPUT_JSON = "10-EVALUATION/llm_ffs/sampled_modelID_glob.json"
MAX_DESC_LEN = 2000


import json, random, statistics
from pathlib import Path

random.seed(55)

def has_features(obj):
    if "Features" not in obj:
        return False
    v = obj["Features"]
    if v is None:
        return False
    if isinstance(v, (dict, list, str)) and len(v) == 0:
        return False
    return True

def get_likes(obj):
    try:
        return int(obj.get("Metadata", {}).get("likes"))
    except (TypeError, ValueError):
        return None

def description_ok(obj):
    desc = obj.get("description")
    if desc is None:
        return True
    if not isinstance(desc, str):
        return True  # or False if you want to be strict
    return len(desc) <= MAX_DESC_LEN

# 1) load + filter eligible files
eligible = []  # list of (path, likes)
for p in FOLDER.rglob("*.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        continue

    model_id = obj.get("modelID")
    if model_id is None:
        continue

    if not has_features(obj):
        continue

    if not description_ok(obj):
        continue

    likes = get_likes(obj)
    if likes is None:
        continue

    eligible.append((p, likes))

if len(eligible) < 500:
    raise RuntimeError(f"Only {len(eligible)} eligible files found (<500).")

# 2) median threshold for stratification
likes_list = [lk for _, lk in eligible]
threshold = statistics.median(likes_list)

popular = [p for p, lk in eligible if lk > threshold]
unpopular = [p for p, lk in eligible if lk <= threshold]

# 3) sample (with fallback if one side is short)
def sample_up_to(items, k):
    return items if len(items) <= k else random.sample(items, k)

pop_sample = sample_up_to(popular, N_PER_GROUP)
unpop_sample = sample_up_to(unpopular, N_PER_GROUP)

need = 2 * N_PER_GROUP - (len(pop_sample) + len(unpop_sample))
if need > 0:
    remaining = list(set(popular + unpopular) - set(pop_sample) - set(unpop_sample))
    filler = sample_up_to(remaining, need)

    # fill whichever group is short first
    pop_short = max(0, N_PER_GROUP - len(pop_sample))
    pop_sample += filler[:pop_short]
    unpop_sample += filler[pop_short:]

sampled_files = pop_sample[:N_PER_GROUP] + unpop_sample[:N_PER_GROUP]
random.shuffle(sampled_files)

# 4) extract model_ids
model_ids = []
for p in sampled_files:
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    model_ids.append(obj["modelID"])

# 5) save JSON
out_path = OUTPUT_JSON
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(model_ids, f, indent=2)

print(f"Eligible files: {len(eligible)}")
print(f"Median likes threshold: {threshold}")
print(f"Popular pool: {len(popular)}, Unpopular pool: {len(unpopular)}")
print(f"Saved {len(model_ids)} model_ids to: {out_path}")
