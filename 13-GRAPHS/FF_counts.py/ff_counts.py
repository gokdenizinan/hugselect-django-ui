import os
import json
from collections import Counter
folder_path = "HF-Models-T7-U"


total_count = 0
unique_features = set()
feature_counter = Counter()
model_feature_counts = []

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            features = data.get("Features", [])
            
            # total + unique
            total_count += len(features)
            unique_features.update(features)
            
            # feature frequency
            feature_counter.update(features)
            
            # model (file) feature count
            model_feature_counts.append((filename, len(features)))

# ---- Results ----
unique_count = len(unique_features)

# Top 10 most common features
top_10_features = feature_counter.most_common(20)

# Models with most features (top 10)
top_10_models = sorted(model_feature_counts, key=lambda x: x[1], reverse=True)[:10]

# ---- Print ----
print("Total feature count:", total_count)
print("Unique feature count:", unique_count)

print("\nTop 10 most common features:")
for feature, count in top_10_features:
    print(f"{feature}: {count}")

print("\nTop 10 models with most features:")
for model, count in top_10_models:
    print(f"{model}: {count}")