import pandas as pd

# G2 VE G5 TEKI ORTAK MODELLER LISTESI
#truthlari nasil cikarmis iki model karsilastrirmak icin
# Load the CSV files
df1 = pd.read_csv("10-EVALUATION/model_ffs_eval_G2.csv")
df2 = pd.read_csv("10-EVALUATION/model_ffs_eval_G5.csv")

# Get unique modelID values from each file
models_1 = set(df1["ModelID"].dropna())
models_2 = set(df2["ModelID"].dropna())

# Find identical modelIDs
common_models = sorted(models_1.intersection(models_2))

# Print results
print("Identical modelID values:")
for model in common_models:
    print(model)
