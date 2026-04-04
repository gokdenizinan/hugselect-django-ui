import json
from collections import defaultdict
from collections import Counter


with open("2-NP_EXTRACTION/NP_full.json", "r", encoding="utf-8") as f:
    np_full = json.load(f)


def ensure_lists(d):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively process nested dictionaries
            new_d[k] = ensure_lists(v)
        elif isinstance(v, list):
            new_d[k] = v  # already a list
        else:
            new_d[k] = [v]  # wrap single value into a list
    return new_d

global_freq = defaultdict(int)
for entry in np_full.values():
    np_name = entry["noun_phrase"]
    global_freq[np_name] += 1
    
# New dictionary for unique sentences
new_dic = {}
np_to_key = {}
seen_sentences = set()
seen_models = set()
seen_authors = set()
seen_nps = set()
counter = 1
low_freq = set()
low_freq_tot = []

for key, entry in np_full.items():
    np = entry["noun_phrase"]

    freq_check = global_freq[entry["noun_phrase"]]

    if freq_check < 10:
        low_freq.add(np)
        low_freq_tot.append(np)
        continue
    
    seen_models.add(entry["model_id"])
    seen_sentences.add(entry["sentence"])
    seen_authors.add(entry["author"])

    if np not in seen_nps:
        new_key = f"NP_{counter}"
        counter += 1
        new_dic[new_key] = ensure_lists(entry)
        new_dic[new_key]["noun_phrase"] = entry["noun_phrase"]
        new_dic[new_key]["np_metadata"]["global_frequency"] = [global_freq[entry["noun_phrase"]]]
        # new_dic[key]["sentence"] = [entry["sentence"]]
        # new_dic[key]["model_id"] = [entry["model_id"]]
        seen_nps.add(np)
        np_to_key[np] = new_key
    else:
        np_key = np_to_key[np]
        if entry["author"] not in new_dic[np_key]["author"]:
            new_dic[np_key]["author"].append(entry["author"])
        if entry["model_id"] not in new_dic[np_key]["model_id"]:
            new_dic[np_key]["model_id"].append(entry["model_id"])
        if entry["sentence"] not in new_dic[np_key]["sentence"]:
            new_dic[np_key]["sentence"].append(entry["sentence"])

        new_dic[np_key]["np_metadata"]["global_frequency"].append(global_freq[entry["noun_phrase"]])
        new_dic[np_key]["np_metadata"]["head_noun"].append(entry["np_metadata"]["head_noun"])
        new_dic[np_key]["np_metadata"]["root"].append(entry["np_metadata"]["root"])
        new_dic[np_key]["np_metadata"]["position_in_text"].append(entry["np_metadata"]["position_in_text"])
        new_dic[np_key]["np_metadata"]["entity_label"].append(entry["np_metadata"]["entity_label"])


for key, entry in new_dic.items():
    my_list = entry["np_metadata"]["global_frequency"]
    entry["np_metadata"]["global_frequency"] = round(sum(my_list) / len(my_list), 1)
    my_list = entry["np_metadata"]["head_noun"]
    entry["np_metadata"]["head_noun"] = max(set(my_list), key=my_list.count)
    my_list = entry["np_metadata"]["root"]
    entry["np_metadata"]["root"] = max(set(my_list), key=my_list.count)
    my_list = entry["np_metadata"]["position_in_text"]
    entry["np_metadata"]["position_in_text"] = round(sum(my_list) / len(my_list),3)
    my_list = entry["np_metadata"]["entity_label"]
    entry["np_metadata"]["entity_label"] = max(set(my_list), key=my_list.count)



print("Total noun phrase size:", len(np_full))
print("Unique noun phrase size:", len(seen_nps))
print("Unique sentences size:", len(seen_sentences))
print("Unique models size:", len(seen_models))
print("###############################")
print("Low frequency (<10) noun phrases:", len(low_freq))
print("Total low frequency noun phrases:", len(low_freq_tot))
# print(Counter(low_freq_tot).most_common(10))



# Limit to first 5 entries for smaller output
# final_dict_small = dict(list(new_dic.items())[2000:3000])

# sorted_data = dict(sorted(new_dic.items(), key=lambda item: len(item[1]['model_id']), reverse=True))

# Save results
with open("2-NP_EXTRACTION/NP_aggregated.json", "w", encoding="utf-8") as f:
    json.dump(new_dic, f, ensure_ascii=False, indent=4)


# # MOST USED SENTENCES
# print(f"Aggregated dictionary saved. Total unique noun phrases: {len(final_dict_small)}")
# for k, v in list(sorted_data.items())[:20]:
#     print(v["sentence"])  
#     print(len(v["model_id"]))
#     print("\n")

import matplotlib.pyplot as plt

# Your data
labels = [
    "Total Noun Phrases",
    "Unique HF Noun Phrases (>=10)",
    "Unique LF Noun Phrases (<10)",
    "Unique Sentences",
    "Unique Models"
]
values = [len(np_full), len(seen_nps), len(low_freq), len(seen_sentences), len(seen_models)]

# Create a bar chart
plt.figure(figsize=(8,5))
bars = plt.bar(labels, values, color=['seagreen', 'mediumseagreen', "mediumspringgreen" ,'c', 'rosybrown'])

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}', ha='center', va='bottom', fontsize=10)

plt.title("Overview of Dataset Sizes")
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("sentence_size.png")
plt.show()
