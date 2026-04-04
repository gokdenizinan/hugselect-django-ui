import requests
import time
import json
from tqdm import tqdm
from itertools import islice
import copy
from collections import defaultdict
from google import genai
import random
import nltk
import random
from nltk.corpus import stopwords
import re
import os

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/split_1.json", "r", encoding = "utf-8") as f:
#     split_1 = json.load(f)

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_input_2.json", "r", encoding = "utf-8") as f:
#     input_2 = json.load(f)

# with open("1-MODEL_FILTERING/model_likes_10k.json", "r", encoding = "utf-8") as f:
#     m_liked = json.load(f)

# with open("1-MODEL_FILTERING/model_likes_10k_Y2.json", "r", encoding = "utf-8") as f:
#     m_liked_Y1 = json.load(f)

# split_list = [entry["model_id"] for entry in split_1]
# input_list = [entry["model_id"] for entry in input_2]


# split_10k = {}
# input_10k = {}
# for key, item in m_liked.items():
#     if key in split_list:
#         split_10k[key] = item
#     if key in input_list:
#         input_10k[key] = item

# print(len(split_list))
# print(len(split_10k))
# print("-----")
# print(len(input_list))
# print(len(input_10k))
# print("-----")


# filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/split_10k.json"
# with open(filename, "w", encoding="utf-8") as f:
#     json.dump(split_10k, f, indent=2, ensure_ascii=False)

# filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/input_10k.json"
# with open(filename, "w", encoding="utf-8") as f:
#     json.dump(input_10k, f, indent=2, ensure_ascii=False)

# diff_keys = set(m_liked.keys()) ^ set(m_liked_Y1.keys())
# print(len(diff_keys))

# filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/model_likes_diff_Y2.json"
# with open(filename, "w", encoding="utf-8") as f:
#     json.dump(list(diff_keys), f, indent=2, ensure_ascii=False)

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
# burasi phase 2 den sonra kontrol amacli kullaniliyor
######

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_input_2.json", "r", encoding = "utf-8") as f:
#     indec = json.load(f)

#     # Sample model_id to find
# target_model_id = "ashwinR/CodeExplainer"

# # Loop through list and get the index
# index_found = None
# for idx, entry in enumerate(indec):
#     if entry.get("model_id") == target_model_id:
#         index_found = idx
#         break

# if index_found is not None:
#     print(f"Model ID '{target_model_id}' found at index: {index_found}")
# else:
#     print(f"Model ID '{target_model_id}' not found in the list.")
def load_mentioned_from_folder(folder_path):
        mentioned = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        m_reddit = data.get("reddit", [])
                        m_stack = data.get("stack", [])
                        if m_stack or m_reddit:
                            if isinstance(data, list):
                                mentioned.extend(data)
                            else:
                                mentioned.append(data)
                    except json.JSONDecodeError:
                        print(f"⚠️ Could not parse {filename}")
        return mentioned

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/check_name_V000.json", "r", encoding = "utf-8") as f:
    yes_no_name = json.load(f)

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/mentioned_reviews_1.json", "r", encoding = "utf-8") as f:
#     mentioned = json.load(f)

MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"
mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)


def get_yes_from_string(yes_output: str):
    yes_dict = {}

    pattern = re.compile(
        r"""
        (?:\d+\.\s*|\*\s*)?           # optional list numbering or bullet
        ['"`]?\*{0,2}                 # optional quote, double quote, or backtick + optional bold before model
        ([\w\-\.\d]+)                  # model name (letters, digits, _, -, .)
        ['"`]?\*{0,2}                 # optional quote, double quote, or backtick + optional bold after model
        \s*[:\-]\s*                    # separator (":" or "-")
        \*{0,2}(Yes|No|Maybe)\*{0,2}  # the answer (ignore bold/quotes/backticks)
        """,
        re.VERBOSE | re.IGNORECASE
    )


    matches = pattern.findall(yes_output)
    for model, answer in matches:
        yes_dict[model.strip()] = answer.capitalize()

    return yes_dict

def mention_check_prompt(yes_no_name):

    no_list = ["NO", "No", "no", " no", " No", "no."]
    yes_list = ["YES", "Yes","yes.","yes", "Yes.", " Yes", "Yes ", "yes "]
    non_matched = []
    yes_maybe_names = {}
    no_names = {}
    for key, item in yes_no_name.items():
        yes_dic = get_yes_from_string(item["output"])
        for model_name in item['list_names']:
            if model_name in item["output"]:
                answer = yes_dic.get(model_name, "Not found")
                if answer not in no_list:
                    if answer in yes_list:
                        yes_maybe_names[model_name] = answer
                    else:
                        yes_maybe_names[model_name] = answer
                else:
                    no_names[model_name] = answer
    
    return yes_maybe_names, no_names

yes_n, no_n = mention_check_prompt(yes_no_name)

filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/yes_name.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(yes_n, f, indent=2, ensure_ascii=False)

filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/no_name.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(no_n, f, indent=2, ensure_ascii=False)


count = 0
yes_no_key = []
for ket, item in yes_no_name.items():
    for i in item["list_names"]:
        yes_no_key.append(i)
        count+=1
print("TOTAL;", count)
print("YES;", len(yes_n))
print("NO;", len(no_n))

keywords = []
for key, item in yes_n.items():
    keywords.append(key)

keywords2 = []
for key, item in no_n.items():
    keywords2.append(key)

keywords_t = keywords + keywords2

for i in yes_no_key:
    if i not in keywords_t:
        print("THIS IS SKIPPED: ", i)

grouped_results = {keyword: [] for keyword in keywords}

# Iterate over data
for entry in mentioned:
    for keyword in keywords:
        if keyword.lower() in entry['topic'].lower():
            grouped_results[keyword].append(entry)

# Optional: Remove empty lists
grouped_results = {k: v for k, v in grouped_results.items() if v}

filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/yes_name_samples.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(grouped_results, f, indent=2, ensure_ascii=False)


# CHECK MENTIONS IN GROUPPED RESULTS:

# Initialize list to store model_ids that are mentioned
def mention_model_ids(grouped_results):
    mentioned_model_ids_reddit = []
    mentioned_model_ids_stack = []


    for name, entries in grouped_results.items():
        for entry in entries:
            model_id = entry.get("model_id", "")
            found = False
            
            # Check in reddit
            for reddit_entry in entry.get("reddit", []):
                for mention in reddit_entry.get("mentioned", []):
                    if model_id in mention:
                        found = True
                        break
                if found:
                    mentioned_model_ids_reddit.append(model_id)
                    break
            
            
            # Check in stack if not found in reddit
            if not found:
                for stack_entry in entry.get("stack", []):
                    for mention in stack_entry.get("mentioned", []):
                        if model_id in mention:
                            found = True
                            break
                    if found:
                        mentioned_model_ids_stack.append(model_id)
                        break
            
            # Add to list if found
    return mentioned_model_ids_reddit, mentioned_model_ids_stack

mentioned_model_ids_reddit, mentioned_model_ids_stack = mention_model_ids(grouped_results)
print("Model IDs mentioned in reddit:")
print(len(mentioned_model_ids_reddit))
men = set(mentioned_model_ids_reddit)
print(len(men))

print("Model IDs mentioned in stack:")
print(len(mentioned_model_ids_stack))
men = set(mentioned_model_ids_stack)
print(len(men))