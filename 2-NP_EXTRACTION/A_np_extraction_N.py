from Ab_generic_terms import GENERIC_TERMS, EXCLUDE, WEAK_HEADS  # Ensure this module is available
import os
import json
import time
import random
from collections import defaultdict, Counter
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
import emoji
import re
DetectorFactory.seed = 0  # makes detection reproducible

ENTITY_TO_CONCEPT = {
    "ORG": "Tool", #
    "PERSON": "Contributor", 
    "NORP": "Language Group", #
    "GPE": "Locale",
    "LOC": "Locale",
    "LANGUAGE": "Language", #
    "PRODUCT": "Model Component", #
    "FAC": "Infrastructure", #
    "WORK_OF_ART": "Dataset", #
    "EVENT": "Benchmark", #
    "LAW": "License", #
    "DATE": "Training Info",
    "TIME": "Training Info",
    "ORDINAL": "Metric",
    "CARDINAL": "Metric",
    "MONEY": "Training Info",
    "PERCENT": "Metric",
    "QUANTITY": "Training Info",
}

USEFUL_CONCEPTS = {
    "Tool", "Dataset", "Model Component", "Infrastructure", "Benchmark", "Language Group", "License", 
}

# USEFUL_CONCEPTS = {
#     "ORG", "NORP", "PRODUCT", "FAC", "WORK_OF_ART", "EVENT", "LAW", 
# }


def map_entity_to_concept(label):
    return ENTITY_TO_CONCEPT.get(label, "Other")

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_trf")

# Start the timer
start_time = time.time()
# Set your directory path
directory = "HF-Models-P9"  # Adjust this path as needed
RANDOM_SEED = 45
random.seed(RANDOM_SEED)

# Load and sample files
all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
SAMPLE = 1 # Use full dataset
sample_size = max(1, int(len(all_files) * SAMPLE))
sampled_files = random.sample(all_files, sample_size)

print(f"Randomly selected {sample_size} of {len(all_files)} files.\n")

# Final result dictionary
np_dict = defaultdict(list)
import re
from spacy.lang.en.stop_words import STOP_WORDS

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

def is_multiword(np, multi_word_only=False):

    if multi_word_only and len(np.split()) <= 1:
        return False

    return True

def is_bad_phrase(sentence):
    global EXCLUDE
    exclude_sentence_prefixes = EXCLUDE
    """
    Check if the noun phrase is part of an excluded phrase in the sentence.
    """
    if any(sentence.lower().startswith(p.lower()) for p in exclude_sentence_prefixes):
        return False
    
    return True


def is_emoji(sentence):

    emoji_count = emoji.emoji_count(sentence)
    # If more than 5 emojis, return False
    if emoji_count > 0:
        return False
    else:
        return True

def process_emojis(sentence):
    # Regex pattern to match all emojis
    cleaned_sentence = emoji.replace_emoji(sentence, "(emoji)")
    return cleaned_sentence


LATIN_REGEX = re.compile(r'^[\u0041-\u024F]+$', re.UNICODE)

def is_english_like(np):
    """
    Return False if any token in the noun phrase is made up entirely 
    of non-Latin characters. Otherwise True.
    """
    
    if any(char in emoji.EMOJI_DATA for char in np):
        return False
    
    tokens = np.split()
    for tok in tokens:
        # Strip punctuation around the token
        tok_clean = re.sub(r'\W+', '', tok)
        if not tok_clean:  # skip empty after stripping
            continue
        # If every character is alphabetic AND it's not Latin → reject
        if tok_clean.isalpha() and not LATIN_REGEX.match(tok_clean):
            return False

    return True



# Extract emojis fr
def extract_noun_phrases(text):
    global filter_counts, concept_counts, filter_counts_set, np_set
    doc = nlp(text)
    entries = []
    ent_span_to_label = {(ent.start, ent.end): ent.label_ for ent in doc.ents}
    key_concepts = list(ENTITY_TO_CONCEPT.keys())

    for chunk in doc.noun_chunks:
        filter_counts["Total"] += 1
        phrase = chunk.text.strip()
        context = chunk.sent.text.strip()
        head_noun = chunk.root.text
        root = chunk.root.pos_
        position = chunk.start / len(doc)  # relative position in doc

        np_key = phrase.lower()
        good_np = 0

        if is_valid_np(phrase):
            filter_counts["Validity"] += 1
            good_np += 1
            if phrase not in np_set:
                filter_counts_set["Validity"] += 1


        if is_structurally_useful_np(chunk):
            filter_counts["Structure"] += 1
            good_np += 1
            if phrase not in np_set:
                filter_counts_set["Structure"] += 1


        entity_label = ent_span_to_label.get((chunk.start, chunk.end), "O") 
        
        if phrase not in np_set:
            if entity_label in key_concepts:
                concept_counts[entity_label] += 1
            else:
                concept_counts["NONE"] += 1
        concept = map_entity_to_concept(entity_label)
        
        if concept in USEFUL_CONCEPTS:
            filter_counts["Concept"] += 1
            good_np += 1
            if phrase not in np_set:
                filter_counts_set["Concept"] += 1

        if is_emoji(context):
            context = process_emojis(context)
        
        if is_english_like(phrase):
            filter_counts["English"] += 1
            good_np += 1
            if phrase not in np_set:
                filter_counts_set["English"] += 1
        
        if phrase not in np_set:
            filter_counts_set["Total"] += 1
        
        if good_np < 4:
            np_set.add(phrase)
            continue  # Skip this NP

        else:
            filter_counts["Passed"] += 1
            if phrase not in np_set:
                np_set.add(phrase)
                filter_counts_set["Passed"] += 1

            entries.append({
                "noun_phrase": phrase,
                "sentence": context,
                "head_noun": head_noun,
                "root": root,
                "position_in_text": round(position, 3),
                "entity_label": entity_label,
                "concept": concept,
                "np_key": np_key  # temp key for counting
            })


    return entries

documents_nps = []  # List of lists of noun phrases per doc
model_ids = []
count_key = 0
np_set = set()

filter_counts = {
    "Total": 0,
    "Validity": 0,
    "Structure": 0,
    "Concept": 0,
    "English": 0,
    "Passed": 0,
}

filter_counts_set = {
    "Total": 0,
    "Validity": 0,
    "Structure": 0,
    "Concept": 0,
    "English": 0,
    "Passed": 0,
}

concept_counts = {
    "NONE": 0,
    "PERSON": 0,
    "GPE": 0,
    "LOC": 0,
    "DATE": 0,
    "TIME": 0,
    "ORDINAL": 0,
    "CARDINAL": 0,
    "MONEY": 0,
    "PERCENT": 0,
    "LANGUAGE": 0, 
    "FAC": 0, #
    "LAW": 0, #
    "QUANTITY": 0,
    "ORG": 0 ,#
    "NORP": 0, #
    "PRODUCT": 0, #
    "WORK_OF_ART": 0, #
    "EVENT": 0, #
}


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

            for np in np_entries:
                # Append to the np_dict with model_id
                count_key += 1
                key = f"NP{count_key}"
                np_dict[key] = {
                    "noun_phrase": np["noun_phrase"],
                    "model_id": model_id,
                    "author": data.get("author", "Unknown"),
                    "model_task": model_task,
                    "sentence": np["sentence"],
                    "np_metadata" : {
                    "head_noun": np["head_noun"],
                    "root": np["root"],
                    "position_in_text": np["position_in_text"],
                    "entity_label": np["entity_label"]
                    }
                }

    except Exception as e:
        print(f"Error reading {filename}: {e}")


# Report
elapsed = time.time() - start_time
print(f"\n✅ Extraction completed in {elapsed:.2f} seconds.")
print(f"Processed {sample_size} models.")
print(f"Extracted {len(np_dict)} noun phrases.")

print("\n📊 Filter Counts:")
for key, count in filter_counts.items():
    print(f"{key}: {count}")

#PLOTTING
labels = list(filter_counts.keys())
counts = list(filter_counts.values())

# Separate passed and failed
failed_labels = labels[1:-1]
failed_counts = counts[1:-1]
passed_label = labels[-1]
passed_count = counts[-1]

plt.figure(figsize=(10, 6))
plt.bar(failed_labels, failed_counts, color='lightseagreen', label='Passed filters')
passed_bar = plt.bar(passed_label, passed_count, color='seagreen', label='Passed all filters')

# Annotate only the passed bar
plt.text(
    x=passed_bar.patches[0].get_x() + passed_bar.patches[0].get_width() / 2,
    y=passed_count,
    s=str(passed_count),
    ha='center',
    va='bottom',
    fontsize=11,
    color='seagreen' 
)

# Reference line for total
total_count = counts [0]# Assuming first element is total
plt.axhline(y=total_count, color='gray', linestyle='--', label='Total sampled files')

plt.ylabel('Number of Noun Phrases')
plt.title('Filtering Criteria')
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right')
ax = plt.gca()  # get current axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('N_np_filter.png', dpi=300)

###################
###################
###################
###################

#PLOTTING np SET
labels = list(filter_counts_set.keys())
counts = list(filter_counts_set.values())

# Separate passed and failed
failed_labels = labels[1:-1]
failed_counts = counts[1:-1]
passed_label = labels[-1]
passed_count = counts[-1]

plt.figure(figsize=(10, 6))
plt.bar(failed_labels, failed_counts, color='lightseagreen', label='Passed filters')
passed_bar = plt.bar(passed_label, passed_count, color='seagreen', label='Passed all filters')

# Annotate only the passed bar
plt.text(
    x=passed_bar.patches[0].get_x() + passed_bar.patches[0].get_width() / 2,
    y=passed_count,
    s=str(passed_count),
    ha='center',
    va='bottom',
    fontsize=11,
    color='seagreen' 
)

# Reference line for total
total_count = counts[0] # Assuming first element is total
plt.axhline(y=total_count, color='gray', linestyle='--', label='Total sampled files')

plt.ylabel('Number of Unique Noun Phrases')
plt.title('Filtering Criteria')
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right')
ax = plt.gca()  # get current axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('N_np_filter_set.png', dpi=300)

###################
###################
###################
###################

#PLOTTING CONCEPT COUNTS
labels_cap = list(concept_counts.keys())
labels = [w.capitalize() for w in labels_cap]

counts = list(concept_counts.values())

indexes_chosen = [i for i, item in enumerate(labels) if map_entity_to_concept(item.upper()) in USEFUL_CONCEPTS]
indexes_rejected = [i for i, item in enumerate(labels) if map_entity_to_concept(item.upper()) not in USEFUL_CONCEPTS]

# Separate passed and failed
new_list = [labels[i] for i in indexes_chosen]
failed_labels = [labels[i] for i in indexes_rejected]
failed_counts = [counts[i] for i in indexes_rejected]
passed_label = [labels[i] for i in indexes_chosen]
passed_count = [counts[i] for i in indexes_chosen]

plt.figure(figsize=(10, 6))
plt.bar(failed_labels, failed_counts, color='lightseagreen', label='Rejected Concept')
passed_bar = plt.bar(passed_label, passed_count, color='seagreen', label='Chosen Concept')


# Reference line for total
total_count = sum(counts)  # Assuming first element is total
plt.axhline(y=total_count, color='gray', linestyle='--', label='Total Sampled NPs')

plt.ylabel('Concept Count')
plt.yscale("log")
plt.title('Concept Filtering Criteria')
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig('N_np_filter_concepts.png', dpi=300)

# Save results
with open("2-NP_EXTRACTION/NP_full_N.json", "w", encoding="utf-8") as out_file:
    json.dump(np_dict, out_file, indent=2)
print("Results saved to NP_deneme.json.")
