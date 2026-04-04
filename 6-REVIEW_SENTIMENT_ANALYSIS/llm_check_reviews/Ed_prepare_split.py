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

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/mentioned_reviews_1.json", "r", encoding = "utf-8") as f:
#     split_1 = json.load(f)

with open("5-REVIEW_COLLECTION/united_reviews/mentioned_reviews_spacy_1.json", "r", encoding = "utf-8") as f:
    split_1 = json.load(f)

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/no_name.json", "r", encoding = "utf-8") as f:
    no_name = json.load(f)

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/yes_name.json", "r", encoding = "utf-8") as f:
#     yes_name = json.load(f)
# NO_NAME SPLIT_1 

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

# yes_n, no_n = mention_check_prompt(yes_no_name)

filtered_dicts = [entry for entry in split_1 if entry["topic"] in no_name.keys()]

print(len(split_1))
print(len(filtered_dicts))

filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/split_1_no_name_spacy.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(filtered_dicts, f, indent=2, ensure_ascii=False)

#### BURASI COMMENTLI OLACAK
# YES_NAME -> NO_NAME SPLIT_1 DUPLICATE KONTROL

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_V1_U.json", "r", encoding = "utf-8") as f:
#     v1 = json.load(f)

# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_V2_U.json", "r", encoding = "utf-8") as f:
#     v2 = json.load(f)

# # Extract model_ids from each
# ids1 = {entry["model_id"] for entry in v1.values()}
# ids2 = {entry["model_id"] for entry in v2.values()}

# # Find overlaps
# common_ids = ids1 & ids2   # intersection


# # Check if there is at least one match
# if common_ids:
#     print("Yes, there are matches.")
#     print(f"Number of matches: {len(common_ids)}")
#     print(f"Matches: {common_ids}")
# else:
#     print("No matches found.")

# # V1 DEN NO NAMELERI CIKAR
# with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/no_name.json", "r", encoding = "utf-8") as f:
#     no_name = json.load(f)

# to_remove = set(no_name.keys())

# filtered_dict1 = {
#     k: v for k, v in v1.items()
#     if v.get("model_id") not in to_remove
# }

# filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_V1_U_filtered.json"
# with open(filename, "w", encoding="utf-8") as f:
#     json.dump(filtered_dict1, f, indent=2, ensure_ascii=False)

# ### UNITE V1 V V2
# united = {}
# united.update(v1)
# united.update(v2)

# filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_reviews/check_name_review_V1_V2_united.json"
# with open(filename, "w", encoding="utf-8") as f:
#     json.dump(united, f, indent=2, ensure_ascii=False)