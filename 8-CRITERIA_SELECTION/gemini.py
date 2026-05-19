import os
import requests
import time
import json
from tqdm import tqdm
import copy
from collections import defaultdict, Counter
from google import genai
import os

# === SAVE LOCATION ===
SAVE_LOC = f"8-CRITERIA_SELECTION/user_intent/"
VERSION = "D"
C_NAME = "testing_gemini"
GEMINI_API_KEY = '' # YENI535


save_nums = list(range(20, 0, -1))
for i in save_nums:
    SAVE_NUM = i
    check_path =f"{SAVE_LOC}checkpoints_{VERSION}_{SAVE_NUM-1}.json"
    if os.path.exists(check_path):
        with open(check_path, "r", encoding = "utf-8") as f:
            checkpoints = json.load(f)
        CHECKPOINT = checkpoints[C_NAME]

        break
    else:
        SAVE_NUM = 0
        CHECKPOINT = 0

print(f"Save num: {SAVE_NUM}/{len(save_nums)}, checkpoint = {CHECKPOINT}")
print("Loading input dict...")

# INPUT
# with open(f"{SAVE_LOC}test_gemini_{VERSION}.json", "r", encoding = "utf-8") as f:
#     sample_dict = json.load(f)

sample_dict = [
  {
    "query_type": "testing",
    "prompt": "hello there!" }]

SAVE_FIL = f"quality_mapping_output_{VERSION}{SAVE_NUM}"
CHECK_LOC = f"checkpoints_{VERSION}{SAVE_NUM}"


client = genai.Client(api_key=GEMINI_API_KEY)


url = 'https://api.together.xyz/v1/chat/completions'


model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Replace with valid model
# model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo" # ucretli olan

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
    checkpoints = {C_NAME: key}
    # name_key = f"{key}_{item["model_id"]}"
    name_key =  name_keys[key-1]

    while retries < 5:
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

                if success >= 1: # eskiden 5ti
                    time.sleep(1) # 0.6 dakikada bir time limit
                else:
                    tqdm.write(f"sleep 10 seconds")
                    time.sleep(10)
                break  # success, exit retry loop
            else:
                success = 0
                tqdm.write(f"Error for key {name_key} with GEMINI: finish_reason {response.candidates[0].finish_reason}")
                hit_error[key][response.candidates[0].finish_reason] += 1

        except Exception as e:
            retries += 1
            tqdm.write(f"Exception for key {name_key} with GEMINI, retry {retries}/5: {e}")
            hit_error[key]["Gemini_Exception"] += 1
            time.sleep(20) # succes-1 li kisim calismiyorsa ciakr            
                
                
        
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
