import pandas as pd
import json

# ---- Load CSV ----

import pandas as pd
import json
import ast

# ---- Load CSV ----
SAMPLE ="T2"
df = pd.read_csv(f"10-EVALUATION/models_ffs/model_ffs_eval_{SAMPLE}.csv")

# ---- Load JSON files ----
with open(f"10-EVALUATION/results_samples/worst_predicted_fp_{SAMPLE}.json", "r") as f:
    worst_predicted = json.load(f)

with open(f"10-EVALUATION/results_samples/worst_truth_fn_{SAMPLE}.json", "r") as f:
    worst_truth = json.load(f)

# ---- Convert string lists to real lists ----
df["Features"] = df["Features"].apply(ast.literal_eval)
df["truth"] = df["truth"].apply(ast.literal_eval)


# ---- Extract feature values ----
worst_predicted_features = {item["feature"] for item in worst_predicted}
worst_truth_features = {item["feature"] for item in worst_truth}

# ---- Remove matching values inside lists ----
df["Features"] = df["Features"].apply(
    lambda lst: [x for x in lst if x not in worst_predicted_features]
)

df["truth"] = df["truth"].apply(
    lambda lst: [x for x in lst if x not in worst_truth_features]
)

# ---- (Optional) Remove rows that became empty ----
df = df[(df["Features"].str.len() > 0) & (df["truth"].str.len() > 0)]

# ---- Save new CSV ----
df.to_csv(f"10-EVALUATION/models_ffs/model_ffs_eval_T{SAMPLE}.csv", index=False)

print(f"Filtered CSV saved as T{SAMPLE}.csv")
