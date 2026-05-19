import requests
import time
import json
from tqdm import tqdm
from itertools import islice
import copy
from collections import defaultdict, Counter
from google import genai
import random
import nltk
import random
from nltk.corpus import stopwords
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import pandas as pd

def select_random_reviews(reviews, n):
    """
    Selects n random reviews from the list of dictionaries.
    If n is larger than the list, returns all reviews shuffled.
    """
    if n >= len(reviews):
        return random.sample(reviews, len(reviews))  # shuffled all
    return random.sample(reviews, n)

def name_check_prompt(random_reviews):

    names_prompts = []
    for rev in random_reviews:
        samp_sen = rev["reviews"]
        model_id = rev["model_id"]
        # rev["prompt"] = f"""Categorize the following review of a foundational model ({model_id}) according to the ISO 25010 quality attributes: Functional Suitability, Performance Efficiency, Interaction Capability, Reliability, Security, Maintainability, Flexibility, Safety
        #                 For each review, output in JSON:
        #                 Primary_Category – the single most relevant attribute
        #                 Secondary_Categories – optional list of others
        #                 Rationale – brief explanation (1 sentence)
        #                 If the review doesn’t clearly match any, output "Unclear".
        #                 Review: "{samp_sen}"
        # """

        rev["prompt"] = f"""
                Categorize the following review of a foundational model ({model_id}) according to the ISO 25010 quality attributes:
                Functional Suitability, Performance Efficiency, Compatibility, Interaction Capability, Reliability, Security, Maintainability, Flexibility, Safety.

                Additionally, determine the overall sentiment (Positive, Neutral, or Negative) expressed in the review.

                For each review, output in JSON:
                - Primary_Category: the single most relevant ISO 25010 attribute
                - Secondary_Categories: optional list of other relevant attributes
                - Rationale: brief explanation (1 sentence)
                - Sentiment: one of ["Positive", "Neutral", "Negative"]

                If the review doesn’t clearly match any category, output "Unclear" for the Primary_Category.

                Review: "{samp_sen}"
                """
        names_prompts.append(rev)
        
    return names_prompts

#  

PHASE = 7
# RUNING LLM WITH PROMPT LISTS
if PHASE == 7:
    SAVE_LOC = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/"
    # VERSION = "A50"
    VERSION = "B50" # en populer 100 haricindekiler

    save_nums = list(range(20, 0, -1))
    for i in save_nums:
        SAVE_NUM = i
        check_path =f"{SAVE_LOC}checkpoints_{VERSION}_{SAVE_NUM-1}.json"
        if os.path.exists(check_path):
            with open(check_path, "r", encoding = "utf-8") as f:
                checkpoints = json.load(f)
            CHECKPOINT = checkpoints["name_review"]

            break
        else:
            SAVE_NUM = 0
            CHECKPOINT = 0
    
    print(f"Save num: {SAVE_NUM}/{len(save_nums)}, checkpoint = {CHECKPOINT}")
    print("Loading input dict...")

    with open(f"{SAVE_LOC}quality_mapping_input_{VERSION}.json", "r", encoding = "utf-8") as f:
        sample_dict = json.load(f)


    SAVE_FIL = f"quality_mapping_output_{VERSION}_{SAVE_NUM}"
    CHECK_LOC = f"checkpoints_{VERSION}_{SAVE_NUM}"
    # CHECKPOINT = 224

    # GEMINI_API_KEY = '' #ESKISI
    # GEMINI_API_KEY = '' # YENISI
    GEMINI_API_KEY = '' # YENI535
    # LLAMA_API_KEY = ''

    client = genai.Client(api_key=GEMINI_API_KEY)


    url = 'https://api.together.xyz/v1/chat/completions'


    headers = {
        'Authorization': f'Bearer {LLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }

    # model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Replace with valid model
    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo" # ucretli olan

    # input_dict = sample_dict

    input_dict = sample_dict[CHECKPOINT:]
    results = {}  # normal dict is enough
    hit_error = defaultdict(lambda: defaultdict(int))

    TURN_LL = True
    STAYPUT = False  # Set to True to stay on LLAMA once switched
    success = 0
    gemini_daily = 0
    key_iteration = 0
    name_keys = [str(i) for i in range(0, len(sample_dict)+2)]

    for key, item in enumerate(tqdm(input_dict, desc="Processing input dict", unit="NP"), start =1):
        prompt = item["prompt"]
        key += CHECKPOINT

        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model_name
        }
        key_iteration += 1
        retries = 0
        run_again = 0
        checkpoints = {"name_review": key}
        # name_key = f"{key}_{item["model_id"]}"
        name_key =  name_keys[key-1]

        while retries < 5:
            if TURN_LL:
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    if response.status_code == 200:
                        description = response.json()['choices'][0]['message']['content'].strip()
                        results[key] = copy.deepcopy(item)
                        results[key]["output"] = description
                        results[key]["model_used"] = "LLAMA"
                        success += 1
                        tqdm.write(f"Success for key {name_key} with LLAMA. Successes in a row: {success}")
                        # Save progress
                        with open(f"{SAVE_LOC + SAVE_FIL}_U.json", "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        # Save Chechpoint
                        with open(f"{SAVE_LOC + CHECK_LOC}.json", "w", encoding="utf-8") as f:
                            json.dump(checkpoints, f, indent=2, ensure_ascii=False)

                        tqdm.write(f"Continuing after {success} consecutive LLAMA successes.")
                        time.sleep(1)
                        break  # success, exit retry loop
                    else:
                        error_json = response.json()
                        error_msg = error_json.get('error', {}).get('message', '').lower()
                        error_code = response.status_code
                        tqdm.write(f"Error for key {name_key} with LLAMA: (code {error_code}), {error_msg}")
                        hit_error[key][f"LLAMA_{error_code}"] += 1
                        if run_again < 3: # change 0 to 3 for more tries when failed
                            run_again += 1
                            time.sleep(min(60 + 2 ** run_again, 120))
                            tqdm.write(f"Retrying for key {name_key} with LLAMA: attempt {run_again}/inf after waiting (120-180s)")
                        else:
                            tqdm.write(f"Switching to next inpu after {run_again} failed attempts")
                            break

                except requests.exceptions.RequestException as e:
                    retries += 1
                    tqdm.write(f"Exception for key {name_key} with LLAMA, retry {retries}/5: {e}")
                    hit_error[key]["LLAMA_Exception"] += 1
                    time.sleep(min(5+3**retries, 60))
                     
                        
                    
            
    # Final save
    with open(f"{SAVE_LOC + SAVE_FIL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(f"{SAVE_LOC + CHECK_LOC}.json", "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, indent=2, ensure_ascii=False)

    with open(f"{SAVE_LOC + SAVE_FIL}_hit_error.json", "w", encoding="utf-8") as f:
        json.dump(hit_error, f, indent=2, ensure_ascii=False)

    print("---------- SUMMARY ----------")
    print("Input sentences processed:", len(sample_dict))
    print("Errors encountered:", len(hit_error))
    print("----------------------------------")
