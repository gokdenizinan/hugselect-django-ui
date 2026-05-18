import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# Load the two CSV files
df1 = pd.read_csv("8-CRITERIA_SELECTION/hits_cluster/experiment_runs_XX_dedup/multisource/experiment_sample_stats_compact.csv")
df2 = pd.read_csv("8-CRITERIA_SELECTION/hits_cluster/experiment_runs_XX_ABL_FF/multisource/experiment_sample_stats_compact.csv")

CHECKING = "recsys_family_root_first_rank" #"recsys_rank" # recsys_family_root_first_rank
# Rename columns to avoid confusion

def rank_to_hit(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.between(1, 10, inclusive="both").astype(int)

df1 = df1[["sample", CHECKING]].rename(columns={CHECKING: "rank_model1"})
df2 = df2[["sample", CHECKING]].rename(columns={CHECKING: "rank_model2"})

df = pd.merge(df1, df2, on="sample")

df["hit_model1"] = rank_to_hit(df["rank_model1"])
df["hit_model2"] = rank_to_hit(df["rank_model2"])

# Merge on sample (IMPORTANT: ensures pairing)
# Drop missing values (if any)

# Ensure binary values
df["hit_model1"] = df["hit_model1"].astype(int)
df["hit_model2"] = df["hit_model2"].astype(int)

# Build contingency table
a = ((df["hit_model1"] == 1) & (df["hit_model2"] == 1)).sum()
b = ((df["hit_model1"] == 1) & (df["hit_model2"] == 0)).sum()
c = ((df["hit_model1"] == 0) & (df["hit_model2"] == 1)).sum()
d = ((df["hit_model1"] == 0) & (df["hit_model2"] == 0)).sum()

table = [[a, b],
         [c, d]]

print("Contingency table:")
print(pd.DataFrame(
    table,
    index=["Model1 = 1", "Model1 = 0"],
    columns=["Model2 = 1", "Model2 = 0"]
))

# Run McNemar test
result = mcnemar(table, exact=True)

print("\nMcNemar Test:")
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")

# Interpretation
alpha = 0.05
if result.pvalue < alpha:
    print("Result: Significant difference between models.")
else:
    print("Result: No significant difference between models.")