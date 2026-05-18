import json
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import copy
import re
import string

# Descriptions Unitedi Da yeniden runla -> NP_ifo 
# with open("3-LLM_FEATURE_EXTRACTION/Descriptions_Z_global_suan_U.json", "r", encoding="utf-8") as f:
#     des_dict_1 = json.load(f)

# with open("3-LLM_FEATURE_EXTRACTION/Descriptions_Z_global_suan_2.json", "r", encoding="utf-8") as f:
#     des_dict_2 = json.load(f)

# des_dict = des_dict_1 | des_dict_2
    
# np_info = copy.deepcopy(des_dict)

# with open("2-NP_EXTRACTION/NP_comfy_GGG.json", "r", encoding="utf-8") as f:
#     np_info = json.load(f)

# # with open("4-LLM_FEATURE_ORGANIZATION/NP_info_go.json", "r", encoding="utf-8") as f:
# #     np_info = json.load(f)

# def get_info(output):
#     # mapping from a/b/c/d → proper names
#     ordered_keys = ["Technical", "Characteristic", "Functionality", "Description"]
#     info = {}

#     output = output.strip()
#     output = output.strip('`')
#     if output.lower().startswith("json"):
#         output = output[4:].strip()

#     try:
#         output = json.loads(output)
#     except json.JSONDecodeError as e:
#         print("Warning: failed to parse JSON:", key, output)
#         return {k: "" for k in ordered_keys}

#     for new_key, old_key in zip(ordered_keys, output.keys()):
#         info[new_key] = output[old_key]

#     return info

# def create_name(dic):
#     name_list = set()
#     author_list = set()
#     modelid_list = set()
#     for key, value in dic.items():
#         ids = value["model_id"]
#         for i in ids:
#             author, name = i.split("__", 1)
#             name_list.add(name)
#             author_list.add(author)
#             modelid_list.add(i)
#     return name_list, author_list, modelid_list
        
# # def check_name(feature, name_list):
# #     return any(feature in name for name in name_list)

# def check_name(feature, name_list):
#     feature = feature.lower()
#     return any(feature == name.lower() for name in name_list)


# # def is_acronym(np: str) -> bool:
# #     return np.isupper() and 2 <= len(np) <= 7

# import re

# def is_acronym(np: str) -> bool:
#     if not (2 <= len(np) <= 7):
#         return False

#     if not np.isalpha():
#         return False

#     # At least 2 uppercase letters
#     if sum(c.isupper() for c in np) < 2:
#         return False

#     # Avoid normal TitleCase words like "Dog"
#     if np.istitle():
#         return False

#     return True



# def contains_expansion(acronym, sentences, description):

#     if isinstance(sentences, str):
#         sentences = [sentences]

#     if not sentences:
#         return []
        
#     expansions = set()
#     for sentence in sentences:
#         initial_words = sentence.split()
#         words = [word.replace('(', '').replace(')', '') for word in initial_words]
#         # words = [word.strip(string.punctuation) for word in initial_words] # FETURE UN ADINDA OLAN PUNCTUAIONI CIKATIYOR OLABILIR
#         n = len(acronym)
        
#         # Iterate through windows of consecutive words
#         for i in range(len(words) - n + 1):
#             initials = "".join(word[0].upper() for word in words[i:i+n] if word)
#             expansion = " ".join(words[i:i+n])
#             if initials == acronym.upper():
#                 expansions.add(expansion.title())

#     if expansions:
#         expansions = expansions
#     else: 
#         expansions = {"(-)"}
#     return expansions, description


# author_list, name_list , modelid_list = create_name(np_info)


# cok_basarili = 0
# basarili = 0
# basarisiz = 0
# kelimeler = []
# for key, value in np_info.items():
#     info = get_info(value["output"])
#     value["info"] = info
#     feature = value["noun_phrase"]

#     if check_name(feature, author_list):
#         value["base_author"] = "yes"
#     else:
#         value["base_author"] = "no"
#     if check_name(feature, name_list): 
#         value["base_model"] = "yes"
#     else:
#         value["base_model"] = "no"
#     if check_name(feature, modelid_list): 
#         value["base_modelID"] = "yes"
#     else:
#         value["base_modelID"] = "no"

#     if is_acronym(feature): # SADELESTIR SONRASINDA
#         expansion_1, _ = contains_expansion(feature, value["sentence"], value["info"]["Description"])
#         expansion_2, description = contains_expansion(feature, value["info"]["Description"] , value["info"]["Description"])
#         if len(expansion_2) == 1 == len(expansion_1):
#             expansion_1 = next(iter(expansion_1))
#             expansion_2 = next(iter(expansion_2))
#             if expansion_1 == expansion_2:
#                 if expansion_1 != "(-)":
#                     np_open = expansion_1 + f" ({feature})"
#                     value["noun_phrase"] = np_open
#                     kelimeler.append(np_open)
#                     cok_basarili += 1
#                 else:
#                     value["noun_phrase"] = feature + "(_)" + f"the description is: {description}"
#                     s = feature + "(_)" + f"the description is: {description}"
#                     kelimeler.append(s)
#                     basarisiz+= 1
#             else:   
#                 if expansion_2 != "(-)":
#                     np_open = expansion_2 + f" ({feature})"
#                     value["noun_phrase"] = np_open
#                     kelimeler.append(np_open)
#                     basarili+=1
#                 else:
#                     if expansion_1 != "(-)":
#                         np_open = expansion_1 + f" ({feature})"
#                         value["noun_phrase"] = np_open
#                         kelimeler.append(np_open)
#                         basarili+=1
#                     else:
#                         value["noun_phrase"] = feature + "(_)" + f"the description is: {description}"
#                         s = feature + "(_)" + f"the description is: {description}"
#                         kelimeler.append(s)
#                         basarisiz+=1
#         else:
#             expansion_all = expansion_2 | expansion_1
#             base = "/".join(sorted(expansion_all)) + f" ({feature})"
#             with_desc = base + f" the description is: {description}"
#             value["noun_phrase"] = with_desc
#             kelimeler.append(with_desc)



# abbreviation_to_llm = {}
# for i, kelime in enumerate(kelimeler):
#     if "(_)" in kelime:
#         abbreviation_to_llm[f"abbreviation_{i}"] = kelime
#     elif "/" in kelime:
#         abbreviation_to_llm[f"abbreviation_{i}"] = kelime

# with open("4-LLM_FEATURE_ORGANIZATION/abbreviation_to_llm_suan.json", "w", encoding="utf-8") as f:
#     json.dump(abbreviation_to_llm, f, indent=2, ensure_ascii=False) 

# # Save the processed data to a new JSON file
# with open("4-LLM_FEATURE_ORGANIZATION/NP_info_global_suan.json", "w", encoding="utf-8") as f:
#     json.dump(np_info, f, indent=2, ensure_ascii=False) 

# print("Processed noun phrases and saved to 'np_info.json'")

# print(f"Total noun phrases processed: {len(np_info)}")
# for i in kelimeler:
#     print(i)

# print("acronyms: ", len(kelimeler))

# print("expansion found twice: ", cok_basarili)
# print("expansion found once: ", basarili)
# print("no expansion : ", basarisiz)


# #PLOTTING 

# # Initialize counters
# keys = ["Technical", "Characteristic", "Functionality"]
# counts = {k: {"yes": 0, "no": 0} for k in keys}

# # Loop over all NP entries
# for np_id, np_data in np_info.items():
#     if "info" in np_data:
#         for k in keys:
#             val = np_data["info"].get(k, "").lower()
#             if val in ["yes", "no"]:
#                 counts[k][val] += 1

# # Prepare data for plotting
# yes_counts = [counts[k]["yes"] for k in keys]
# no_counts = [counts[k]["no"] for k in keys]

# # Plot stacked bar chart
# x = range(len(keys))
# plt.bar(x, yes_counts, label="Yes", color="seagreen")
# plt.bar(x, no_counts, bottom=yes_counts, label="No", color="indianred")

# plt.xticks(x, keys)
# plt.ylabel("Count")
# plt.title("Feature Classification")
# plt.savefig("technical_yes_no.png")
# plt.legend()
# plt.show()

# ######### UNITED TECHINAL YES NO AND MODEL USED BASE MOEDL
# import matplotlib.pyplot as plt
# from collections import Counter
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch


# # Define color maps
# colors_first3 = {
#     "yes": "seagreen",
#     "no": "indianred"
# }
# colors_last2 = {
#     "yes": "teal",
#     "no": "goldenrod"
# }

# keys = ["Technical", "Characteristic", "Functionality"]
# counts = {k: {"yes": 0, "no": 0} for k in keys}

# # Loop over all NP entries
# for np_id, np_data in np_info.items():
#     if "info" in np_data:
#         for k in keys:
#             val = np_data["info"].get(k, "").lower()
#             if val in ["yes", "no"]:
#                 counts[k][val] += 1

# bases_m = [np_data.get("base_model") for np_data in np_info.values() if "base_model" in np_data]
# bases_a = [np_data.get("base_author") for np_data in np_info.values() if "base_author" in np_data]

# base_m_counts = Counter(bases_m)
# base_a_counts = Counter(bases_a)

# # Plot stacked bars
# fig, ax = plt.subplots(figsize=(10, 6))

# for key in keys:
#     bottom = 0
#     for val in ["yes", "no"]:  # ensure consistent order
#         count = counts[key][val]
#         ax.bar(key, count, bottom=bottom, color=colors_first3.get(val, "gray"))
#         bottom += count

# # Fourth column (Base Model)
# bottom = 0
# for base, count in base_m_counts.items():
#     ax.bar("Base Model", count, bottom=bottom, label=base, color=colors_last2.get(base, "gray"))
#     bottom += count

# # Fifth column (Base Author)
# bottom = 0
# for base, count in base_a_counts.items():
#     ax.bar("Base Author", count, bottom=bottom, label=base, color=colors_last2.get(base, "gray"))
#     bottom += count

# # Labels and title
# ax.set_ylabel("Count of NPs")

# # ax.set_title("Stacked Columns: Model Used vs Base Author & Model")

# # Legend (remove duplicates)
# # Legend using custom handles
# # Legend: show yes/no colors for each column type
# legend_elements = [
#     Patch(facecolor=colors_first3["yes"],  label="Model Character: Yes"),
#     Patch(facecolor=colors_first3["no"], label="Model Character: No"),
#     Patch(facecolor=colors_last2["yes"], label="Model Type: Yes"),
#     Patch(facecolor=colors_last2["no"], label="Model Type: No")
# ]
# ax.legend(handles=legend_elements, title="Column Type / Value", loc="upper right")

# # Save and show plot
# plt.savefig("N_np_general.png")
# plt.show()


# # TECHNICAL YES NO VISUALISATION

# # Initialize counters
# categories = {
#     "Three" : 0,
#     "Two" : 0,
#     "Two (Technical icl)" : 0,
#     "One" : 0,
#     "One (Technical)" : 0,
#     "Zero" : 0
# }

# keys = ["Technical", "Characteristic", "Functionality"]

# # Classify each NP
# for np_data in np_info.values():
#     info = np_data.get("info", {})
#     yes_keys = [k for k in keys if info.get(k, "").lower() == "yes"]
    
#     if len(yes_keys) == 3:
#         categories["Three"] += 1
#     elif len(yes_keys) == 2:
#         categories["Two"] += 1
#         if yes_keys[0]=="Technical" or yes_keys[1] == "Technical":
#             categories["Two (Technical icl)"] += 1
#     elif len(yes_keys) == 1:
#         if yes_keys[0] == "Technical":
#             categories["One (Technical)"] += 1
#         categories["One"] += 1
#     else:
#         categories["Zero"] += 1

# # Plot
# plt.bar(categories.keys(), categories.values(), color="chocolate")
# plt.ylabel("Number of Features")
# plt.title("Distribution of Feature Info Classification")
# plt.xticks(rotation=15, ha="right")  # rotate 45 degrees and align right
# plt.savefig("N_info_yes_no")
# plt.show()



# # PLOTTING MODEL USED
# colors = {
#     "GEMINI": "gold",
#     "LLAMA": "teal",
#     "yes": "seagreen",
#     "no": "indianred"
# }

# models = [np_data.get("model_used") for np_data in np_info.values() if "model_used" in np_data]
# bases_m = [np_data.get("base_model") for np_data in np_info.values() if "base_model" in np_data]
# bases_a = [np_data.get("base_author") for np_data in np_info.values() if "base_author" in np_data]

# # Count frequencies
# model_counts = Counter(models)
# base_m_counts = Counter(bases_m)
# base_a_counts = Counter(bases_a)


# # Plot stacked bars
# fig, ax = plt.subplots()

# # First column (model_used)
# bottom = 0
# for model, count in model_counts.items():
#     ax.bar("Model Used", count, bottom=bottom, label=model, color=colors.get(model, "gray"))
#     bottom += count

# # Second column (base_model)
# bottom = 0
# for base, count in base_m_counts.items():
#     ax.bar("Base Model", count, bottom=bottom, label=base, color=colors.get(base, "gray"))
#     bottom += count

# # Second column (base_model)
# bottom = 0
# for base, count in base_a_counts.items():
#     ax.bar("Base Author", count, bottom=bottom, label=base, color=colors.get(base, "gray"))
#     bottom += count

# ax.set_ylabel("Count of NPs")
# ax.set_title("Stacked Columns: Model Used vs Base Author & Model")
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), title="Category", loc="upper right")
# plt.savefig("N_model_used.png")
# plt.show()


# ###############
# ###############
# ###############
# # EXTRA PLOT
# ###############
# ###############
# ###############

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast

# Load CSV
df = pd.read_csv("10-EVALUATION/models_ffs/model_ffs_new.csv")  # adjust the path if needed

# Convert stringified lists to actual lists
df['Features'] = df['Features'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Count how many models each feature appears in
feature_counter = Counter()
for features in df['Features']:
    feature_counter.update(features)
print(f"Total number of unique features: {len(feature_counter)}")

# Count how many features have the same frequency
freq_of_freq = Counter(feature_counter.values())

# Prepare data for scatter
x = sorted(freq_of_freq.keys())        # number of models a feature appears in
y = [freq_of_freq[i] for i in x]       # number of features with that frequency

# Assign groups based on x
colors = []
groups = []
for xi in x:
    if xi < 10:           # less than 10 models
        colors.append('palevioletred')
        groups.append('Under 10 Models')
    elif xi < 100:        # 10–99 models
        colors.append('sienna')
        groups.append('10 to 99 Models')
    else:                 # >= 100 models
        colors.append('rebeccapurple')
        groups.append('Over 100 Models')

# Plot scatter with log x-axis
plt.figure(figsize=(10,6))
for color, group in set(zip(colors, groups)):
    # Plot only points in this group
    xs = [xi for xi, g in zip(x, groups) if g == group]
    ys = [yi for yi, g in zip(y, groups) if g == group]
    plt.scatter(xs, ys, color=color, s=50, marker='x', alpha=0.7, label=group)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Models per Feature (log scale)")
plt.ylabel("Number of Features (log scale)")
# plt.title("Feature Frequency Across Models")
# plt.grid(True, linestyle="--", alpha=0.5, which='both')
plt.legend(title="Feature Frequency")
plt.tight_layout()
plt.savefig("N_feature_model_dist_scatter_seg_log.png", dpi=300)
plt.show()

