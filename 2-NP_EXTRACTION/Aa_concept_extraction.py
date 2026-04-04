from Ab_generic_terms import GENERIC_TERMS  # Ensure this module is available
import os
import json
import time
import random
from collections import defaultdict, Counter
import spacy
from tqdm import tqdm

ENTITY_TO_CONCEPT = {
    "ORG": "Tool",
    "PERSON": "Contributor",
    "NORP": "Language Group",
    "GPE": "Locale",
    "LOC": "Locale",
    "LANGUAGE": "Language",
    "PRODUCT": "Model Component",
    "FAC": "Other",
    "WORK_OF_ART": "Dataset",
    "EVENT": "Benchmark",
    "LAW": "License",
    "DATE": "Training Info",
    "TIME": "Training Info",
    "ORDINAL": "Metric",
    "CARDINAL": "Metric",
    "MONEY": "Training Info",
    "PERCENT": "Metric",
    "QUANTITY": "Training Info",
}

USEFUL_CONCEPTS = {
    "Tool", "Language", "Dataset", "Model Component", "Benchmark"
}

WEAK_HEADS = {"thing", "one", "stuff", "something", "anything", "aspect", "type"}

useless_concepts = 0

def map_entity_to_concept(label):
    return ENTITY_TO_CONCEPT.get(label, "Other")

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_trf")

# Start the timer
start_time = time.time()
# Set your directory path
directory = "HF-Models-Y3"  # Adjust this path as needed
RANDOM_SEED = 22
random.seed(RANDOM_SEED)

# Load and sample files
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
SAMPLE = 0.1  # Use full dataset
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files.\n")

# Final result dictionary
np_dict = defaultdict(list)
import re
from spacy.lang.en.stop_words import STOP_WORDS

def is_valid_np(np):
    np = np.strip().lower()
    if re.fullmatch(r"\W+", np): # Remove if only punctuation
        return False
    if np in STOP_WORDS: # Remove if just a stopword like "this", "those"
        return False
    if np.lower() in GENERIC_TERMS: # Remove if it’s a generic term
        return False
    if len(np.split()) == 1 and len(np) < 3:# Remove if it’s just a single short word
        return False
    if np.isdigit():  # Remove if it’s just a number
        return False
    if any(np.startswith(determiner + " ") for determiner in {"the", "this", "that", "these", "those", "a", "an"}):
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
    return True

def is_multiword(np, sen, multi_word_only=False, exclude_sentence_prefixes=None):
    """
    Checks whether a noun phrase entry passes optional filters.

    Args:
        entry (dict): The noun phrase entry.
        multi_word_only (bool): If True, filters out single-word noun phrases.
        exclude_sentence_prefixes (list): List of phrases that, if the sentence starts with them, cause exclusion.

    Returns:
        bool: True if entry is valid, False otherwise.
    """
    exclude_sentence_prefixes = exclude_sentence_prefixes or []

    np = np
    sentence = sen

    if multi_word_only and len(np.split()) <= 1:
        return False

    if any(sentence.lower().startswith(p.lower()) for p in exclude_sentence_prefixes):
        return False

    return True


def extract_noun_phrases(text):
    global useless_concepts
    doc = nlp(text)
    temp_entries = []
    ent_span_to_label = {(ent.start, ent.end): ent.label_ for ent in doc.ents}

        
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        np_key = phrase.lower()

        if is_valid_np(phrase) and is_structurally_useful_np(chunk) and is_multiword(phrase, chunk.sent.text.strip(), multi_word_only=True, exclude_sentence_prefixes=None):
            context = chunk.sent.text.strip()
            head_noun = chunk.root.text
            root = chunk.root.pos_
            position = chunk.start / len(doc)  # relative position in doc
            entity_label = ent_span_to_label.get((chunk.start, chunk.end), "O")  # "O" = not a named entity

            concept = map_entity_to_concept(entity_label)
            if concept in USEFUL_CONCEPTS:

                temp_entries.append({
                    "noun_phrase": phrase,
                    "sentence": context,
                    "head_noun": head_noun,
                    "root": root,
                    "position_in_text": position,
                    "entity_label": entity_label,
                    "concept": concept,
                    "np_key": np_key  # temp key for counting
                })

    # Calculate frequency per NP (case insensitive)
    frequencies = Counter(entry["np_key"] for entry in temp_entries)

    # Assign frequency to each NP entry
    noun_phrases = []
    for entry in temp_entries:
        entry["frequency"] = frequencies[entry["np_key"]]
        del entry["np_key"]  # Clean up
        noun_phrases.append(entry)

    return noun_phrases

documents_nps = []  # List of lists of noun phrases per doc
model_ids = []
# Track collected counts per entity label
entity_label_counts = defaultdict(int)
MAX_PER_LABEL = 10

count_key = 0

# Main loop
for filename in tqdm(sampled_files):
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get("description", "")
            model_id = data.get("modelId", filename.replace(".json", ""))
            model_task = data.get("pipeline_tag", "unknown")

            np_entries = extract_noun_phrases(text)

            documents_nps.append(" ".join([np["noun_phrase"] for np in np_entries]))
            # Create a unique key for this model_id and ta
            
            for np in np_entries:
                label = np["entity_label"]
                count_key += 1
                # Skip if we already have enough for this label
                if entity_label_counts[label] >= MAX_PER_LABEL:
                    continue

                definition = f"Definition placeholder for '{np['noun_phrase']}'"
                key = f"NP{count_key}"
                np_dict[key] = {
                    "noun_phrase": np["noun_phrase"],
                    "model_id": model_id,
                    "sentence": np["sentence"],
                    "frequency": np["frequency"],
                    "model_task": model_task,
                    "definition": definition,
                    "head_noun": np["head_noun"],
                    "root": np["root"],
                    "position_in_text": np["position_in_text"],
                    "entity_label": np["entity_label"],
                    "concept": np["concept"]
                }

                # Increase count for that label
                entity_label_counts[label] += 1

    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Optional: print summary of how many we got per label
print("\n📊 Collected NP counts per entity label:")
for label, count in entity_label_counts.items():
    print(f"{label}: {count}")

with open("2-NP_EXTRACTION/NP100.json", "w", encoding="utf-8") as out_file:
    json.dump(np_dict, out_file, indent=2)
print("Results saved.")