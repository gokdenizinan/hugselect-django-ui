import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("8-CRITERIA_SELECTION/hits_cluster/experiment_runs_XX_dedup/multisource/experiment_sample_stats_compact.csv")

# =========================
# 2. Define columns
# =========================
normal_cols = {
    "recsys": "recsys_rank",
    "chatgpt": "chatgpt_rank",
    "gemini": "gemini_rank",
    "claude": "claude_rank",
    "perplexity": "perplexity_rank",
}

family_cols = {
    "recsys": "recsys_family_root_first_rank",
    "chatgpt": "chatgpt_family_root_first_rank",
    "gemini": "gemini_family_root_first_rank",
    "claude": "claude_family_root_first_rank",
    "perplexity": "perplexity_family_root_first_rank",
}

# =========================
# 3. Convert rank -> binary hit@10
#    1 if rank is between 1 and 10
#    0 otherwise (including NaN)
# =========================
def rank_to_hit(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.between(1, 10, inclusive="both").astype(int)

for model, col in normal_cols.items():
    df[f"{model}_normal_hit"] = rank_to_hit(df[col])

for model, col in family_cols.items():
    df[f"{model}_family_hit"] = rank_to_hit(df[col])

# =========================
# 4. McNemar helper
# =========================
def run_mcnemar(dataframe, col_a, col_b, label_a, label_b, exact=True):
    a = ((dataframe[col_a] == 1) & (dataframe[col_b] == 1)).sum()
    b = ((dataframe[col_a] == 1) & (dataframe[col_b] == 0)).sum()
    c = ((dataframe[col_a] == 0) & (dataframe[col_b] == 1)).sum()
    d = ((dataframe[col_a] == 0) & (dataframe[col_b] == 0)).sum()

    table = [[a, b],
             [c, d]]

    result = mcnemar(table, exact=exact)

    return {
        "model_a": label_a,
        "model_b": label_b,
        "a_both_1": a,
        "b_a1_b0": b,
        "c_a0_b1": c,
        "d_both_0": d,
        "statistic": result.statistic,
        "p_value": result.pvalue,
    }

# =========================
# 5. Compare recsys vs others
# =========================
results = []

other_models = ["chatgpt", "gemini", "claude", "perplexity"]

# Normal comparisons
for other in other_models:
    results.append(
        {
            "type": "normal",
            **run_mcnemar(
                df,
                "recsys_normal_hit",
                f"{other}_normal_hit",
                "recsys",
                other
            )
        }
    )

# Family comparisons
for other in other_models:
    results.append(
        {
            "type": "family",
            **run_mcnemar(
                df,
                "recsys_family_hit",
                f"{other}_family_hit",
                "recsys",
                other
            )
        }
    )

results_df = pd.DataFrame(results)

# =========================
# 6. Add significance label
# =========================
alpha = 0.05
results_df["significant"] = results_df["p_value"] < alpha

# Optional: reorder columns
results_df = results_df[
    [
        "type",
        "model_a",
        "model_b",
        "a_both_1",
        "b_a1_b0",
        "c_a0_b1",
        "d_both_0",
        "statistic",
        "p_value",
        "significant",
    ]
]

# =========================
# 7. Show and save results
# =========================
print("\nMcNemar test results:")
print(results_df.to_string(index=False))

results_df.to_csv("mcnemar_results.csv", index=False)
print("\nSaved results to mcnemar_results.csv")