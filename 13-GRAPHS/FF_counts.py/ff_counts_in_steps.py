import json

# Load your JSON file
with open("2-NP_EXTRACTION/NP_X4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect unique noun phrases
unique_noun_phrases = set()

# Sum counts
total_count = 0

for entry in data.values():
    noun_phrase = entry.get("noun_phrase")
    count = entry.get("count", 0)

    if noun_phrase:
        unique_noun_phrases.add(noun_phrase)
    
    total_count += count

# Results
num_unique_noun_phrases = len(unique_noun_phrases)

print("Unique noun phrases:", num_unique_noun_phrases)
print("Total count:", total_count)