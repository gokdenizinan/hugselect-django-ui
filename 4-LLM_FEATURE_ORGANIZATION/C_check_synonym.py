import nltk
import json
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

# Make sure you have these downloaded first:
# nltk.download('wordnet')
# nltk.download('omw-1.4')
import pandas as pd

# 1️⃣ Load the CSV
df = pd.read_csv("4-LLM_FEATURE_ORGANIZATION/df_feature_info_global.csv")  # adjust the path if needed

# 2️⃣ Filter rows where base_model and base_author are "no"
filtered_df = df[(df["Base Model"] == "no") & (df["Base Author"] == "no")]

# 3️⃣ Get the list of 'features' values
features = filtered_df["Feature"].tolist()

# 4️⃣ Optional: print the list
print(len(features))


# Example list of features

# ---- Part 1: Check WordNet synonyms ----
def get_wordnet_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

wordnet_matches = {}
for feature in features:
    syns = get_wordnet_synonyms(feature)
    overlaps = [f for f in features if f != feature and f in syns]
    if overlaps:
        wordnet_matches[feature] = overlaps

# ---- Part 2: Semantic similarity with embeddings ----
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(features, convert_to_tensor=True)
sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

semantic_matches = {}
threshold = 0.8  # tune this for stricter/looser matching
for i in range(len(features)):
    for j in range(i+1, len(features)):
        score = sim_matrix[i][j].item()
        if score >= threshold:
            semantic_matches.setdefault(features[i], []).append((features[j], round(score, 2)))

# ---- Results ----
print("WordNet-based synonym overlaps:")
print(wordnet_matches)

# Save semantic_matches as JSON
with open("4-LLM_FEATURE_ORGANIZATION/synonym_matches_united_glob.json", "w", encoding="utf-8") as f:
    json.dump(semantic_matches, f, indent=2, ensure_ascii=False)

print("✅ semantic_matches saved to semantic_matches.json")
