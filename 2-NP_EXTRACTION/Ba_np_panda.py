import pandas as pd
from collections import Counter, defaultdict
import json

with open("2-NP_EXTRACTION/NP_full_no_emojis.json", "r", encoding="utf-8") as f:
    np_full = json.load(f)

# Prepare data structures
np_summary = defaultdict(lambda: {
    "occurrences": 0,
    "models": set(),
    "sentences": set(),
    "tasks": []
})

# Populate the summary
for entry in np_full.values():
    np = entry["noun_phrase"]
    np_summary[np]["occurrences"] += 1
    np_summary[np]["models"].add(entry["model_id"])
    np_summary[np]["tasks"].append(entry["model_task"])
    sentence = entry.get("sentence", "")
    np_summary[np]["sentences"].add(sentence)



# Prepare rows for the DataFrame
rows = []
for np, info in np_summary.items():
    occ = info["occurrences"]
    unique_models = len(info["models"])
    
    # Count tasks
    task_counter = Counter(info["tasks"])
    most_common_task, task_count = task_counter.most_common(1)[0]
    
    # Percentage of occurrence in the most common task
    task_percentage = (task_count / occ) * 100 if occ > 0 else 0
    
    rows.append({
        "Noun Phrase": np,
        "Occurrences": occ,
        "Unique Models": unique_models,
        "Unique Sentences": len(info["sentences"]),
        "Most Common Task": most_common_task,
        "Task %": round(task_percentage, 2)
    })

# Create the DataFrame
df = pd.DataFrame(rows)

# Sort by Occurrences descending
df = df.sort_values(by="Occurrences", ascending=False).reset_index(drop=True)

# Display
print(df.head(20))  # top 20 for readability
# Collect all sentences into a set (removes duplicates automatically)
unique_sentences = {entry["sentence"] for entry in np_full.values()}

print(f"Total number of unique Sentences: {len(unique_sentences)}")

total_occurrences = df["Unique Sentences"].sum()
print("Total Sentences:", total_occurrences)

total_unque_models = df["Occurrences"].shape[0]
print("Total Unique NPS:", total_unque_models) 


# Assume df is your DataFrame
df.to_csv("2-NP_EXTRACTION/noun_phrases_summary_noemoji.csv", index=False, encoding="utf-8")
