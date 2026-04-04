# import A_np_extraction_N
import json
from collections import defaultdict
from collections import Counter


with open("2-NP_EXTRACTION/NP_full_N.json", "r", encoding="utf-8") as f:
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
seen_sentences_lf = set()
seen_sentences = set()

seen_models = set()
seen_models_lf = set()

seen_authors = set()
seen_nps = set()
counter = 1
low_freq = set()
normal_np = set()
low_freq_tot = []
high_freq = set()
high_freq_tot =[]

for key, entry in np_full.items():
    np = entry["noun_phrase"]

    freq_check = global_freq[entry["noun_phrase"]]

    if freq_check < 10:
        low_freq.add(np)
        low_freq_tot.append(np)
        seen_sentences_lf.add(entry["sentence"])
        seen_models_lf.add(entry["model_id"])

        continue

    normal_np.add(np)
    high_freq_tot.append(np)
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
print("Low frequency (<10) noun phrases:", len(normal_np))
print("Total low frequency noun phrases:", len(high_freq_tot))
# print(Counter(low_freq_tot).most_common(10))



# Limit to first 5 entries for smaller output
# final_dict_small = dict(list(new_dic.items())[2000:3000])

# sorted_data = dict(sorted(new_dic.items(), key=lambda item: len(item[1]['model_id']), reverse=True))

# # Save results
# with open("2-NP_EXTRACTION/NP_aggregated_N.json", "w", encoding="utf-8") as f:
#     json.dump(new_dic, f, ensure_ascii=False, indent=4)



# #BU PLOTU ayni plot icinde UCE AYIR -> 6 column olcak sekilde
# #BIRINCISI: total unique np, hf uniqu,
# # total unique sentence HF unique sentence,
# # total models, total HF Models

# import matplotlib.pyplot as plt


# # Counts
# print(len(normal_np))
# print(len(seen_nps))
# total_unique_nps = len(seen_nps)+len(low_freq)                    # all NPs seen
# hf_nps = len(seen_nps)                           # high-frequency NPs (>=10)
# lf_nps = len(low_freq)                             # low-frequency NPs (<10)

# total_unique_sentences = len(seen_sentences) + len(seen_sentences_lf)
# hf_np_sentences = len(seen_sentences)
# lf_np_sentences =  len(seen_sentences_lf)   # low-freq NPs total sentences

# unique_models = len(seen_models) + len(seen_models_lf)
# hf_np_models = len(seen_models)
# lf_np_models = len(seen_models_lf)

# # Your data
# labels = [
#     # "Unique Noun Phrases",
#     "HF Noun Phrases",
#     "LF Noun Phrases",
#     # "Unique Sentences",
#     "HF NP Sentences",
#     "LF NP Sentences",
#     # "Unique Models",
#     "HF NP Models",
#     "LF NP Models",
# ]


# counts = [
#     # total_unique_nps,
#     hf_nps,
#     lf_nps,
#     # total_unique_sentences,
#     hf_np_sentences,
#     lf_np_sentences,
#     # unique_models,
#     hf_np_models,
#     lf_np_models,
# ]


# # Create a bar chart
# plt.figure(figsize=(10,6))
# bars = plt.bar(labels, counts, color= ['mediumseagreen', "mediumseagreen",
#           'c', 'c', 'rosybrown', 'rosybrown'])

# # Add value labels on top of bars
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}', ha='center', va='bottom', fontsize=10)

# # plt.title("Overview of NP Extraction")
# from matplotlib.patches import Patch

# # Add horizontal lines for totals
# line1 = plt.axhline(y=total_unique_nps, color='mediumseagreen', linestyle='-.', label='Unique Noun Phrases')
# line2 = plt.axhline(y=total_unique_sentences, color='c', linestyle='-.', label='Unique Sentences')
# line3 = plt.axhline(y=unique_models, color='rosybrown', linestyle='-.', label='Unique Models')

# # Legend elements for bar groups
# legend_elements = [
#     Patch(facecolor='mediumseagreen', label='Noun Phrases'),
#     Patch(facecolor='c', label='Sentences'),
#     Patch(facecolor='rosybrown', label='Models')
# ]

# # Combine bar group patches and line handles
# all_handles = legend_elements + [line1, line2, line3]
# plt.legend(handles=all_handles, loc='upper right')

# plt.ylabel("Count")
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.xticks(rotation=30,ha='right')
# plt.tight_layout()
# plt.savefig("N_np_extraction_summary.png")
# plt.show()