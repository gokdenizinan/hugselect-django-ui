import pandas as pd
import ast

# Load CSV
path ="10-EVALUATION/models_ffs/model_ffs_eval_TT5.csv"

df = pd.read_csv(path)

# Convert string representation of lists into actual Python lists
df["Features"] = df["Features"].apply(ast.literal_eval)
df["truth"] = df["truth"].apply(ast.literal_eval)

# --------------------------------------------------
# 1️⃣ Total possible feature vocabulary
# (Union of all features appearing in either column)
# --------------------------------------------------
all_features = set()

for features in df["Features"]:
    all_features.update(features)

for truth in df["truth"]:
    all_features.update(truth)

total_vocabulary_size = len(all_features)

# --------------------------------------------------
# 2️⃣ Average number of true features per model
# --------------------------------------------------
avg_true_features = df["truth"].apply(len).mean()

# --------------------------------------------------
# 3️⃣ Average number of predicted features per model
# --------------------------------------------------
avg_predicted_features = df["Features"].apply(len).mean()

# --------------------------------------------------
# Print results
# --------------------------------------------------
print("Total possible feature vocabulary:", total_vocabulary_size)
print("Average number of true features per model:", avg_true_features)
print("Average number of predicted features per model:", avg_predicted_features)