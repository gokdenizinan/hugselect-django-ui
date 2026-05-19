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

PHASE = 0
print("doing stuff...")

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


if PHASE == 1: 
    # with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/mentioned_reviews_1.json", "r", encoding = "utf-8") as f:
    #     mentioned = json.load(f) #bunu original yap ve yeni mentioned icin path ekle

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)

    def get_model_names(mentioned):
        model_names = set()
        for model in mentioned: 
            model_names.add(model["topic"])
        return list(model_names)

    names = get_model_names(mentioned)

    def filter_new_mentions(new, original):
        original_names = get_model_names(original)
        new_names = get_model_names(new)
        unique_names = [n for n in new_names if n not in original_names]
        for name in new_names:
            if name in original_names:
                print(f"old model found: {name}")
        return unique_names
    
    # names = filter_new_mentions(mentioned, original)

    def name_check_prompt(names):
        names_prompts = []
        base_prompt = (
            "I have a list of reviews collected from reddit and stackoverflow that mention "
            "the name of some foundational models. "
            "If I gave you the name of the foundational model, can you tell me whether it is possible "
            "that the reviews don’t refer to the foundational model but something else? "
            "For each model, answer with a simple: Yes, No or Maybe."
        )

        chunk_size = 100
        num_chunks = len(names) // chunk_size + (1 if len(names) % chunk_size != 0 else 0)

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            list_names = names[start:end]

            names_and_prompts = {
                "list_names": list_names,
                "prompt": f"{base_prompt} The foundational models are: {list_names}"
            }
            names_prompts.append(names_and_prompts)

        return names_prompts
    
    #BURA ONEMLI
    sample_dict = name_check_prompt(names)

    filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/check_name_input.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sample_dict, f, indent=2, ensure_ascii=False)

# INCOMPLETE FOR CHECKING MODELS WITH REVIEWS

if PHASE == 2:
    with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/check_name_V000.json", "r", encoding = "utf-8") as f:
        yes_no_name = json.load(f)

    # with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/mentioned_reviews_1.json", "r", encoding = "utf-8") as f:
    #     mentioned = json.load(f)

    MENTIONED_FOLDER = "5-REVIEW_COLLECTION/united_mentioned_reviews"
    mentioned = load_mentioned_from_folder(MENTIONED_FOLDER)

    print("2")
    def first_100_words(text):
        words = text.split()
        # Take up to 50 words
        truncated_words = words[:50]
        result = " ".join(truncated_words)
        # Enforce character limit
        return result[:400]

    stop_words = set(stopwords.words("english"))
    def preprocess(sentence):
        words = [w.strip(".,!?") for w in sentence.lower().split()]
        return set(w for w in words if w and w not in stop_words)

    def jaccard_similarity(sent1, sent2):
        set1, set2 = preprocess(sent1), preprocess(sent2)
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    def deduplicate_to_groups(sentences, threshold=0.7, k=5):
        groups = []
        used = set()
        
        for i, sent in enumerate(sentences):
            if i in used:
                continue  # already grouped
            
            group = [sent]
            used.add(i)
            
            for j in range(i+1, len(sentences)):
                if j not in used and jaccard_similarity(sent, sentences[j]) >= threshold:
                    group.append(sentences[j])
                    used.add(j)
            
            groups.append(group)
        
        # pick one random representative per group
        representatives = [random.choice(group) for group in groups]

        # adjust to exactly k representatives
        if len(representatives) > k:
            representatives = random.sample(representatives, k)  # pick k groups
        elif len(representatives) < k:
            # add extra reps from groups until we have k
            while len(representatives) < k:
                group = random.choice(groups)
                representatives.append(random.choice(group))

        return representatives, groups

    def get_simple_reviews(reviews):
        five_reviews, _ = deduplicate_to_groups(reviews)
        fin_reviews = []
        for item in five_reviews:
            fin_reviews.append(first_100_words(item))
        return fin_reviews
    
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

    def mention_model_ids(grouped_results):
        mentioned_model_ids_reddit = []
        mentioned_model_ids_stack = []


        for entry in grouped_results:
        
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

    def mention_check_prompt(yes_no_name, mentioned):
        base_prompt = (
            "I have a list of reviews that mention a foundational model. "
            "I want you to determine whether these reviews actually refer to the foundational model. "
            "Answer with a simple Yes, No or Maybe. "
        )
        mentioned_reddit, mentioned_stack = mention_model_ids(mentioned)
        no_list = ["NO", "No", "no", " no", " No", "no."]
        non_matched = []
        yes_names = []
        for key, item in yes_no_name.items():
            yes_dic = get_yes_from_string(item["output"])
            for model_name in item['list_names']:
                if model_name in item["output"]:
                    answer = yes_dic.get(model_name, "Not found")
                    # if answer == "Not found":
                    #     raise ValueError("Model not found")
                    if answer not in no_list:
                        yes_names.append(model_name)
        rewievs_prompt = []

        for model_name in tqdm(yes_names): 
            #BURA COK ONEMLI!!! MODEL_ID MI TOPIC MI LOOPLANCAK BEYNIM YANDI
            # Find the first matching review
            matches = [item for item in mentioned if item["topic"] == model_name]
            if len(matches) == 1:
                match = matches[0]
                reddit_mentions = [mention for reddit_entry in match["reddit"] for mention in reddit_entry["mentioned"]]
                stack_mentions = [mention for stack_entry in match["stack"] for mention in stack_entry["mentioned"]]

            elif len(matches) > 1:
                reddit_mentions = []
                stack_mentions = []               # Loop over each reddit entry
                for matc in matches:
                    for reddit_entry in matc["reddit"]:
                        for mention_text in reddit_entry.get("mentioned", []):
                            reddit_mentions.append(mention_text)
                    for stack_entry in matc["stack"]:
                        for mention_text in stack_entry.get("mentioned", []):
                            stack_mentions.append(mention_text)
            else:
                non_matched.append(model_name)

            if match:
                if match.get("reddit") and reddit_mentions:
                    if match["model_id"] not in mentioned_reddit:
                        revs_and_prompts_reddit = {
                            "model_id": match["model_id"],
                            "topic" : match["topic"],
                            "reddit" : match["reddit"], 
                            "prompt": f"{base_prompt} The sentences for the foundational model {match["topic"]} are: {get_simple_reviews(reddit_mentions)}"
                        }
                        rewievs_prompt.append(revs_and_prompts_reddit)


                if match.get("stack") and stack_mentions:
                    if match["model_id"] not in mentioned_stack:
                        revs_and_prompts_stack = {
                            "model_id": match["model_id"],
                            "topic" : match["topic"],
                            "stack" : match["stack"], 
                            "prompt": f"{base_prompt} The sentences for the foundational model {match["topic"]} are: {get_simple_reviews(stack_mentions)}"
                        }
                        rewievs_prompt.append(revs_and_prompts_stack)
        
        return rewievs_prompt, non_matched
    print("3")

    #BURA ONEMLI
    sample_dict, non_matched = mention_check_prompt(yes_no_name, mentioned)
    print("4")

    for i in non_matched:
        print(i)

    print(len(sample_dict))

    filename = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/check_name_review.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sample_dict, f, indent=2, ensure_ascii=False)


PHASE = 8
# RUNING LLM WITH PROMPT LISTS
if PHASE == 0:
    SAVE_LOC = f"6-REVIEW_SENTIMENT_ANALYSIS/llm_check_full/"

    save_nums = list(range(20, 0, -1))
    for i in save_nums:
        SAVE_NUM = i
        check_path =f"{SAVE_LOC}checkpoints_V{SAVE_NUM-1}.json"
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

    with open(f"{SAVE_LOC}check_name_input.json", "r", encoding = "utf-8") as f:
        sample_dict = json.load(f)


    SAVE_FIL = f"check_name_input_V{SAVE_NUM}"
    CHECK_LOC = f"checkpoints_V{SAVE_NUM}"
    # CHECKPOINT = 224

    GEMINI_API_KEY = ''
    LLAMA_API_KEY = ''

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

print("done!")