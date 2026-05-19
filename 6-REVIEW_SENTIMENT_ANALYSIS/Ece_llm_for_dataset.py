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
        samp_sen = rev["processed"]
        model_id = rev["model_id"]
        rev["prompt"] = f"""I will give you a review sentence about the foundational model {model_id}.
        Grade the meaningfulness of the sentence from 0 to 10 considering the following: - pseudo_perplexity (fluency of the text), - nsp_coherence (logical coherence of consecutive sentences), - topic_stability (consistency of the topic).
        Return the grade as a JSON like this: {{ "grade": x }}. Do not include any other text.
        The sentence is: {samp_sen}
        """
        names_prompts.append(rev)
        
    return names_prompts

with open(f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json", "r", encoding = "utf-8") as f:
    filtered = json.load(f)

random_reviews = select_random_reviews(filtered, 500) # SAMPLE IN CASE YOU WANT TO TRY IT FIRST

sample_dict = name_check_prompt(random_reviews)

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/check_meaning_input.json", "w", encoding = "utf-8") as f:
    json.dump(sample_dict, f , ensure_ascii=False, indent=2)

PHASE = 7
# RUNING LLM WITH PROMPT LISTS
if PHASE == 7:
    SAVE_LOC = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/"

    save_nums = list(range(20, 0, -1))
    for i in save_nums:
        SAVE_NUM = i
        check_path =f"{SAVE_LOC}checkpoints_V5{SAVE_NUM-1}.json"
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

    with open(f"{SAVE_LOC}check_meaning_input.json", "r", encoding = "utf-8") as f:
        sample_dict = json.load(f)


    SAVE_FIL = f"check_meaning_output_V5{SAVE_NUM}"
    CHECK_LOC = f"checkpoints_V5{SAVE_NUM}"
    # CHECKPOINT = 224

    GEMINI_API_KEY = ''
    LLAMA_API_KEY = '' # benim original

    client = genai.Client(api_key=GEMINI_API_KEY)


    url = 'https://api.together.xyz/v1/chat/completions'


    headers = {
        'Authorization': f'Bearer {LLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }

    model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Replace with valid model

    # input_dict = sample_dict

    input_dict = sample_dict[CHECKPOINT:]
    results = {}  # normal dict is enough
    hit_error = defaultdict(lambda: defaultdict(int))

    TURN_LL = True
    STAYPUT = False  # Set to True to stay on LLAMA once switched
    success = 0
    gemini_daily = 0
    key_iteration = 0
    name_keys = [str(i) for i in range(0, len(input_dict)+2)]

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
                        
                        if success >= 1 and not STAYPUT: # change 1 to for more tries when succeeded
                            TURN_LL = False
                            tqdm.write(f"Switching to GEMINI after {success} consecutive LLAMA successes.")
                            success = 0
                        elif key_iteration % 50 == 0 and STAYPUT:
                            STAYPUT = False
                            tqdm.write("Setting STAYPUT to True after 100 iterations.")
                        else:
                            tqdm.write(f"Continuing with LLAMA. after waiting (120-140s).")
                            time.sleep(min(120+2**success, 140)) 
                        break  # success, exit retry loop
                    else:
                        success = 0
                        error_json = response.json()
                        error_msg = error_json.get('error', {}).get('message', '').lower()
                        error_code = response.status_code
                        tqdm.write(f"Error for key {name_key} with LLAMA: (code {error_code}), {error_msg}")
                        hit_error[key][f"LLAMA_{error_code}"] += 1
                        if STAYPUT or run_again < 0: # change 0 to 3 for more tries when failed
                            run_again += 1
                            time.sleep(min(120 + 2 ** run_again, 180))
                            tqdm.write(f"Retrying for key {name_key} with LLAMA: attempt {run_again}/inf after waiting (120-180s)")
                            if key_iteration % 50 == 0:
                                STAYPUT = False
                                tqdm.write("Resetting STAYPUT to False after 100 iterations.")
                        else:
                            success = 0
                            TURN_LL = False
                            tqdm.write(f"Switching to GEMINI after LLAMA failure.")

                except requests.exceptions.RequestException as e:
                    retries += 1
                    tqdm.write(f"Exception for key {name_key} with LLAMA, retry {retries}/5: {e}")
                    hit_error[key]["LLAMA_Exception"] += 1
                    time.sleep(min(5+3**retries, 60))
                    if retries >= 1 and not STAYPUT:
                        success = 0
                        TURN_LL = False
                        
            else:
                try:
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    if response.candidates[0].finish_reason == "STOP":
                        description = response.text
                        results[key] = copy.deepcopy(item)
                        results[key]["output"] = str(description)
                        results[key]["model_used"] = "GEMINI"
                        success += 1
                        gemini_daily = 0
                        tqdm.write(f"Success for key {name_key} with GEMINI. Successes in a row: {success}")
                        # Save progress
                        with open(f"{SAVE_LOC + SAVE_FIL}_U.json", "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                        with open(f"{SAVE_LOC + CHECK_LOC}.json", "w", encoding="utf-8") as f:
                            json.dump(checkpoints, f, indent=2, ensure_ascii=False)
    
                        if success >= 5: #5 or 6 check to see how long it takes to switch to LLAMA (ideal is 120-140 seconds)
                            TURN_LL = True
                            tqdm.write(f"Switching to LLAMA after {success} consecutive GEMINI successes.") 
                            success = 0
                        else:
                            time.sleep(min(14+2**success, 60))
                        break  # success, exit retry loop
                    else:
                        success = 0
                        tqdm.write(f"Error for key {name_key} with GEMINI: finish_reason {response.candidates[0].finish_reason}")
                        hit_error[key][response.candidates[0].finish_reason] += 1
                        TURN_LL = True

                except Exception as e:
                    retries += 1
                    tqdm.write(f"Exception for key {name_key} with GEMINI, retry {retries}/5: {e}")
                    hit_error[key]["Gemini_Exception"] += 1
                    time.sleep(min(120+2**retries, 180)) # succes-1 li kisim calismiyorsa ciakr
                    gemini_daily += 1
                    if retries >= 1: 
                        success = 0
                        TURN_LL = True
                        if gemini_daily >= 2:
                            tqdm.write("Reached daily limit for GEMINI API calls. Stopping GEMINI usage.")
                            STAYPUT = True
                        
                    
            
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
