import pandas as pd

# File paths
file7 = "11-RECOMMENDATION_EVALUATION/paper_model_2/snipped_papers_8.csv"
file8 = "11-RECOMMENDATION_EVALUATION/MORE_PAPERS/snipped_papers_8.csv"

# Function to auto-detect delimiter
def read_csv_auto(file):
    try:
        return pd.read_csv(file, sep=";")
    except:
        return pd.read_csv(file, sep=",")

# Read files
df7 = read_csv_auto(file7)
df8 = read_csv_auto(file8)

# Ensure both have same columns (add missing ones)
for col in df7.columns:
    if col not in df8.columns:
        df8[col] = None

for col in df8.columns:
    if col not in df7.columns:
        df7[col] = None

# Align column order
df8 = df8[df7.columns]

# Find duplicates based on title
duplicates = df8[df8["title"].isin(df7["title"])]

# Print skipped duplicates
if not duplicates.empty:
    print("Skipped duplicates (title):")
    print(duplicates["title"].tolist())

print(len(duplicates), "duplicates skipped based on title.")

# Remove duplicates from df8
df8_filtered = df8[~df8["title"].isin(df7["title"])]

# Combine data
combined_df = pd.concat([df7, df8_filtered], ignore_index=True)

# Save result
combined_df.to_csv("11-RECOMMENDATION_EVALUATION/MORE_PAPERS/merged.csv", index=False)

print("Merged file saved as merged.csv")