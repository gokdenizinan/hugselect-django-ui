import json
import os
from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS  # Ensure this module is available
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import requests
from tqdm import tqdm
import copy
import time
from collections import defaultdict, Counter
from google import genai


print("doing stuff...")
sample_queries = ["Best open-source LLM for general instruction following with ≤10B parameters",
                  "Foundation vision transformer models for image classification pretraining",
                  "Multilingual text embedding models optimized for semantic search and retrieval",
                  "State-of-the-art open-weight speech-to-text foundational models with streaming support" ,
                  "Lightweight transformer models for on-device NLP inference",
                  "Open-weight language models optimized for code generation and debugging assistance",
                  "Multimodal foundational models that support image + text understanding and reasoning",
                  "Diffusion-based text-to-image models with high fidelity and open licensing",
                  "Large text embedding models trained for reranking or dense retrieval tasks",
                  "LLMs trained for long-context reasoning (≥128k context length)",
                  "Foundational models specialized for biomedical or clinical NLP tasks",
                  "Open-source audio classification / audio embedding foundational models",
                  "Machine translation foundational models supporting low-resource languages",
                  "Compact diffusion or transformer models for real-time image generation on consumer GPUs (<8GB VRAM)",
                  "Reinforcement learning–tuned conversational models optimized for safety and alignment"]


# === HELPER FUNCTIONS ===
def is_valid_np(np):
    global EXCLUDE
    np = np.strip().lower()
    if re.fullmatch(r"\W+", np): # Remove if only punctuation
        return False
    if np in STOP_WORDS: # Remove if just a stopword like "this", "those"
        return False
    if np.lower() in GENERIC_TERMS: # Remove if it’s a generic term
        return False
    if len(np.split()) == 1 and len(np) < 3: # Remove if it’s just a single short word
        return False
    if np.isdigit():  # Remove if it’s just a number
        return False
    if any(np.startswith(determiner + " ") for determiner in {"the", "this", "that", "these", "those", "a", "an"}):
        return False
     
    exclude_sentence_prefixes = EXCLUDE
    if any(np.lower().startswith(p.lower()) for p in exclude_sentence_prefixes): #np.lower mi sentence mi kontrol et bu functionu
        return False
    return True

def is_structurally_useful_np(chunk):
    # Rule 1: All tokens are adjectives/adverbs (e.g., "fast modern")
    if all(token.pos_ in {"ADJ", "ADV"} for token in chunk):
        return False
    # Rule 2: Adjective modifying another adjective (e.g., "custom powerful new")
    for token in chunk: 
        if token.dep_ == "amod" and token.head.pos_ == "ADJ":
            return False
    # Rule 3: Short NP with ambiguous head noun (e.g., "that one", "this thing")
    if len(chunk) <= 2 and chunk.root.lemma_.lower() in WEAK_HEADS:
        return False

    noun_exist = any(token.pos_ in {"NOUN", "PROPN", "NUM"} for token in chunk)
    if not noun_exist:
        return False
    
    return True

nlp = spacy.load("en_core_web_sm")
def extract_noun_phrases(text):
    doc = nlp(text)
    entries = []
    ent_span_to_label = {(ent.start, ent.end): ent.label_ for ent in doc.ents}
        
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        context = chunk.sent.text.strip()
        head_noun = chunk.root.text
        root = chunk.root.pos_
        position = chunk.start / len(doc)  # relative position in doc

        np_key = phrase.lower()
        good_np = 0
        if is_valid_np(phrase):
            good_np += 1


        if is_structurally_useful_np(chunk):
            good_np += 1

        if good_np == 2:
            entries.append(phrase)

    return entries

##3 === Feature Matching ===
import os
import json

def iterate_feature_matches(folder_path, sample_features):
    sample_set = set(sample_features)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        full_path = os.path.join(folder_path, filename)

        # Load only one dictionary at a time
        with open(full_path, "r") as f:
            data = json.load(f)

        model_id = data.get("modelId")
        if model_id is None:
            continue

        features = data.get("Features", [])
        matched = [f for f in features if f in sample_set]

        if matched:
            yield model_id, {
                "matched_features": matched,
                "num_matches": len(matched),
                "searched_features": sample_features,
            }

PHASE = 0
if PHASE == 2 :
    sample_dict = []
    def quality_prompt(query):
        samp_prompt = f"""
            Classify the user’s Hugging Face model search query into one or more of the following attributes:
            Functional Suitability
            Performance Efficiency
            Compatibility
            Interaction Capability
            Reliability
            Security
            Maintainability
            Flexibility
            Safety
            Instructions:
            Identify which attributes the query is asking about or implying.
            Only choose attributes clearly supported by the query.
            Do not provide an explanation.
            The Query is : {query}
            """
        return samp_prompt

    for q in sample_queries:
        sample_dict.append({
            "query": q,
            "prompt": quality_prompt(q)
        })

    with open("8-CRITERIA_SELECTION/query/quality_query_input_Q15.json", "w") as f:
        json.dump(sample_dict, f, indent=4)

PHASE = 0
if PHASE == 3 :

    sample_dict = []
    def quality_prompt(query):
        samp_prompt = f"""
            You are given a user query about a Hugging Face model.
            Extract as much structured metadata as possible from the query and return it in the JSON schema below.

            Rules:
            - Only fill fields that are explicitly mentioned or very clearly implied by the query.
            - For anything missing or uncertain, use null, 0, false, empty lists [], or empty objects {{}} as appropriate.
            - Do not invent, guess, or hallucinate values.
            - Output valid JSON only, no extra text.

            Schema to fill:
            {{
            "modelId": null,
            "author": null,
            "description": null,
            "Features": [],
            "metadata": {{
                "license": null,
                "usedStorage": null,
                "inferenceProviderMapping": null,
                "resourceGroup": null,
                "last_modified": null,
                "downloads_last_30_days": null,
                "disabled": null,
                "model_index": null,
                "inference": null,
                "library_name": null,
                "private": null,
                "tensors": null,
                "lastModified": null,
                "metrics": null,
                "likes": null,
                "cardData": {{
                "pipeline_tag": null,
                "tags": []
                }},
                "widgetData": null,
                "downloads_all_time": null,
                "xetEnabled": null,
                "baseModels": null,
                "summary": null,
                "pipeline_tag": null,
                "gguf": null,
                "language": null,
                "childrenModelCount": {{
                "adapter": 0,
                "merge": 0,
                "quantized": 0,
                "finetune": 0
                }},
                "tags": [],
                "datasets": null,
                "transformersInfo": {{
                "auto_model": null,
                "custom_class": null,
                "pipeline_tag": null,
                "processor": null
                }},
                "mask_token": null,
                "trendingScore": null,
                "gated": null,
                "file_count": null,
                "url": null,
                "features": null,
                "config": {{
                "architectures": [],
                "model_type": null,
                "tokenizer_config": {{
                    "cls_token": null,
                    "mask_token": null,
                    "pad_token": null,
                    "sep_token": null,
                    "unk_token": null
                }}
                }},
                "spaces": []
            }}
            }}

            The Query is: {query}
            """
        return samp_prompt


    for q in sample_queries:
        sample_dict.append({
            "query": q,
            "prompt": quality_prompt(q)
        })

    with open("8-CRITERIA_SELECTION/query/metadata_query_input_Q16.json", "w") as f:
        json.dump(sample_dict, f, indent=4)


PHASE = 0
if PHASE == 4 :

    SAVE_LOC = f"8-CRITERIA_SELECTION/query/"
    VERSION = "Q16"
    # VERSION = "B50" # en populer 100 haricindekiler

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

    with open(f"{SAVE_LOC}metadata_query_input_{VERSION}.json", "r", encoding = "utf-8") as f:
        sample_dict = json.load(f)


    SAVE_FIL = f"metadata_query_output_{VERSION}_{SAVE_NUM}"
    CHECK_LOC = f"checkpoints_{VERSION}_{SAVE_NUM}"
    # CHECKPOINT = 224

    # GEMINI_API_KEY = '' #ESKISI
    # GEMINI_API_KEY = '' # YENISI
    GEMINI_API_KEY = '' # YENI535
    # LLAMA_API_KEY = ''
    LLAMA_API_KEY = "" # HOCANIN VERDIGI

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



import json
import re

with open(f"8-CRITERIA_SELECTION/query/metadata_query_output_Q16_0.json", "r", encoding = "utf-8") as f:
    metadata_q = json.load(f)


def extract_non_null(d):
    result = {}

    if isinstance(d, dict):
        for key, value in d.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                nested = extract_non_null(value)
                if nested:  
                    result[key] = nested
            else:
                result[key] = value

    elif isinstance(d, list):
        items = []
        for item in d:
            if item is None:
                continue
            if isinstance(item, (dict, list)):
                nested = extract_non_null(item)
                if nested:
                    items.append(nested)
            else:
                items.append(item)
        return items

    return result
for a, data in metadata_q.items():
    print("========== ====== === = === =====")
    raw = data["output"]

    # remove code fences like ```json ... ```
    clean = re.sub(r"```.*?\n|```$", "", raw, flags=re.DOTALL).strip()

    q_indv = json.loads(clean)
    non_null = extract_non_null(q_indv)
    print(json.dumps(non_null, indent=4))


PHASE = 0
if PHASE == 99:
    folder_path = "/Users/ardacanseradali/Documents/Thesis_master/HF-Models-T3"

    query_results = []
    for sam_q in sample_queries:
        sam_l = sam_q.lower()
        np_entries = extract_noun_phrases(sam_l)
        print(np_entries)
        result = {}
        for model_id, match_info in iterate_feature_matches(folder_path, np_entries):
            result[model_id] = match_info
        query_results.append(result)
        
    with open("8-CRITERIA_SELECTION/feature_query.json", "w") as f:
        json.dump(result, f, indent=4)

    print("done")




# import pandas as pd

# # Load your CSV file
# df = pd.read_csv("4-LLM_FEATURE_ORGANIZATION/df_feature_info.csv")   # ← replace with your file path

# # Check for matches (case-insensitive)
# matches = df[df['Feature'].str.contains("image classification", case=True, na=False)]

# print(matches)
