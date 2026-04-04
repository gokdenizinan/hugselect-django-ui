import json
import csv
from collections import Counter

papers_path = "11-RECOMMENDATION_EVALUATION/OUTPUT_ZZZZ_VVV.json"
likes_path = "1-MODEL_FILTERING/N_sorted_model_likes_P9.json"
output_path = "11-RECOMMENDATION_EVALUATION/OUTPUT_ZZZZ_VVVVV.json"
csv_path = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/merged_2.csv"
model_to_year_path = "11-RECOMMENDATION_EVALUATION/model_to_year_more.json"

# --------------------------------------------------
# Choose the output key order here
# Any keys not listed here will be appended at the end
# in their original order.asfdasf
# --------------------------------------------------
desired_order = [
    "model_full_name",
    "source",
    "in approach",
    "manuel_check",
    "likes",
    "year",
    "model_year",
    "task",
    "domain",
    "selection_rationale",
    "user_intent",
    "evidence",
    "title",
    "rank",
]

ALLOWED_LETTER_RANKS = {"AA", "A", "Q1", "Q2"}


def is_valid_venue(value):
    """Return True if venue is AA/A/Q1/Q2 or an integer >= 100."""
    if value is None:
        return False

    v = str(value).strip()
    if not v:
        return False

    if v in ALLOWED_LETTER_RANKS:
        return True

    try:
        return int(v) >= 100
    except ValueError:
        return False


def reorder_dict(d, desired_order):
    """Return a new dict with keys in desired_order first, then remaining keys."""
    ordered = {}

    for key in desired_order:
        if key in d:
            ordered[key] = d[key]

    for key, value in d.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


with open(model_to_year_path, "r", encoding="utf-8") as f:
    model_to_year = json.load(f)

with open(papers_path, "r", encoding="utf-8") as f:
    papers = json.load(f)

with open(likes_path, "r", encoding="utf-8") as f:
    model_likes = json.load(f)

# build mapping from CSV: sample -> metadata
sample_info = {}
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        sample = row["sample"]
        sample_info[sample] = {
            "source": row.get("saved_name", ""),
            "year": row.get("year", ""),
            "title": row.get("title", ""),
            "rank": row.get("venue", ""),   # rename venue -> rank
            "model_full_name": row.get("matched_hf_model", ""),
        }

updated = {}

for old_key, paper_data in papers.items():
    parts = old_key.split("_")
    if len(parts) < 2:
        # if key format is unexpected, skip or set null
        updated[old_key] = None
        continue

    number = parts[1]
    new_key = f"sample_{number}"

    # preserve nulls already present in papers
    if paper_data is None:
        updated[new_key] = None
        continue

    # Rule 1: sample number must exist in CSV
    info = sample_info.get(new_key)
    if info is None:
        updated[new_key] = None
        continue

    # Rule 2: venue must be valid
    rank_value = info.get("rank", "")
    if not is_valid_venue(rank_value):
        updated[new_key] = None
        continue

    new_dict = paper_data.copy()

    new_dict["source"] = info.get("source", "")
    new_dict["year"] = info.get("year", "")
    new_dict["title"] = info.get("title", "")
    new_dict["rank"] = rank_value
    new_dict["model_full_name"] = info.get("model_full_name", "")

    model_name = new_dict.get("model_full_name", "").strip()

    if "manuel_check" not in new_dict or new_dict["manuel_check"] is None:
        new_dict["manuel_check"] = ""

    new_dict["likes"] = model_likes.get(model_name, 0)

    model_year_val = model_to_year.get(model_name)
    new_dict["model_year"] = "" if model_year_val is None else str(model_year_val)

    updated[new_key] = new_dict

organized = {}
for paper_id, paper_data in updated.items():
    if paper_data is None:
        organized[paper_id] = None
    else:
        organized[paper_id] = reorder_dict(paper_data, desired_order)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(organized, f, indent=2, ensure_ascii=False)

# ------------------------
# stats
# ------------------------
total_samples = len(organized)
non_null_samples = sum(1 for v in organized.values() if v is not None)

with_model_year = sum(
    1 for v in organized.values()
    if isinstance(v, dict) and v.get("model_year") not in (None, "", "None")
)

approach_counter = Counter()
for v in organized.values():
    if isinstance(v, dict):
        val = v.get("in approach")
        if val is not None:
            approach_counter[val] += 1

print("\n===== STATS =====")
print(f"Total samples: {total_samples}")
print(f"Non-null samples: {non_null_samples}")
print(f"Samples with model_year: {with_model_year}")

print("\nDistribution of 'in approach':")
for k, v in approach_counter.items():
    print(f"{k}: {v}")