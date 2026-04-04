import json
from collections import Counter, defaultdict

# Load your JSON file
input_path = "11-RECOMMENDATION_EVALUATION/OUTPUT_ZZ.json" # your input JSON file
output_path = "11-RECOMMENDATION_EVALUATION/OUTPUT_analysis.json" # output JSON file for results

with open(input_path, "r") as f:
    data = json.load(f)

# Initialize containers
dist_in_approach = Counter()
dist_manual_check = Counter()
dist_model_year = Counter()
dist_year = Counter()
dist_domain = Counter()
dist_task = Counter()

mismatch_samples = []
likes_candidates = []
model_name_counts = Counter()

# Process data
for sample_name, info in data.items():
    if not isinstance(info, dict):
        continue
    # Distributions
    dist_in_approach[info.get("in approach")] += 1
    dist_manual_check[info.get("manuel_check")] += 1
    dist_model_year[info.get("model_year")] += 1
    dist_year[info.get("year")] += 1
    dist_domain[info.get("domain")] += 1
    dist_task[info.get("task")] += 1

    # Model name counts
    model_name = info.get("model_full_name")
    if model_name:
        model_name_counts[model_name] += 1

    # Year mismatch check
    if info.get("model_year") != info.get("year"):
        mismatch_samples.append(sample_name)

    # Likes filtering
    if info.get("manuel_check") in ["yes", "maybe"]:
        likes_candidates.append((sample_name, info.get("likes", 0)))

# Lowest 5 likes
lowest_5_likes = sorted(likes_candidates, key=lambda x: x[1])[:5]

# Models appearing more than once
duplicate_models = {
    model: count
    for model, count in model_name_counts.items()
    if count > 1
}

# Final result
result = {
    "distributions": {
        "in_approach": dict(dist_in_approach),
        "manuel_check": dict(dist_manual_check),
        "model_year": dict(dist_model_year),
        "year": dict(dist_year),
        "domain": dict(dist_domain),
        "task": dict(dist_task),
    },
    "year_mismatches": {
        "count": len(mismatch_samples),
        "samples": mismatch_samples
    },
    "lowest_5_likes_manual_check_yes_maybe": [
        {"sample": s, "likes": l} for s, l in lowest_5_likes
    ],
    "duplicate_models": duplicate_models
}

# Save to JSON
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Analysis saved to {output_path}")