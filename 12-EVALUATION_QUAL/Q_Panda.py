from collections import Counter
import pandas as pd
import json
# Example: your dictionary of dictionaries
# data = {...}

with open("6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/quality_mapping_output_AB50_all_expanded.json", "r") as f:
    data = json.load(f)


def create_attribute_sample_csv(data_dict, sample_size=500, output_path="attribute_sample.csv", random_state=42):
    """
    Creates a proportional stratified sample based on Primary_Category
    and saves it as a CSV file.

    Parameters:
    - data_dict: dictionary of dictionaries
    - sample_size: total number of samples
    - output_path: output CSV filename
    - random_state: for reproducibility
    """

    # Convert dict-of-dicts to DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient="index").reset_index()
    df = df.drop(columns=["index"])  # drop the original keys if not needed

    # Keep only required columns
    df = df[["model_id", "reviews", "Primary_Category", "Rationale"]]

    # Remove rows with missing category
    # df = df.dropna(subset=["Primary_Category"])

    # Compute proportional allocation
    proportions = df["Primary_Category"].value_counts(normalize=True)
    sample_counts = (proportions * sample_size).round().astype(int)

    # Fix rounding difference
    diff = sample_size - sample_counts.sum()
    if diff != 0:
        largest_class = sample_counts.idxmax()
        sample_counts[largest_class] += diff

    # Perform stratified sampling
    sampled_df = (
        df.groupby("Primary_Category", group_keys=False)
          .apply(lambda x: x.sample(n=sample_counts[x.name], random_state=random_state)
                                                                .assign(Primary_Category=x.name))
    )

    # Shuffle final dataset
    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Save to CSV
    sampled_df.to_csv(output_path, index=False)

    print(f"Sample of {len(sampled_df)} saved to {output_path}")

    return sampled_df

sample_df = create_attribute_sample_csv(data, sample_size=500, output_path="12-EVALUATION_QUAL/attribute_sample_3.csv", random_state=42)


# import json
# import pandas as pd

# IN_CSV = "12-EVALUATION_QUAL/attribute_sample.csv"
# MAPPING_JSON = "6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/preprocessing_sentiment.json"  # <-- change this
# OUT_CSV = "12-EVALUATION_QUAL/attribute_sample_2.csv"

# # 1) load sampled/output csv
# df = pd.read_csv(IN_CSV)

# # 2) load mapping json (list of dicts)
# with open(MAPPING_JSON, "r", encoding="utf-8") as f:
#     mapping = json.load(f)

# map_df = pd.DataFrame(mapping)

# # sanity: keep only needed columns & drop duplicates
# map_df = map_df[["model_id", "processed", "original"]].dropna()
# map_df = map_df.drop_duplicates(subset=["model_id", "processed"], keep="first")

# # 3) figure out which column in df contains the processed review text
# # change this to your actual column name in the csv:
# PROCESSED_COL = "reviews"  # <-- change if needed (e.g. "text", "content", etc.)

# if PROCESSED_COL not in df.columns:
#     raise ValueError(f"Couldn't find '{PROCESSED_COL}' in CSV columns: {list(df.columns)}")

# # 4) merge
# out = df.merge(
#     map_df,
#     how="left",
#     left_on=["model_id", PROCESSED_COL],
#     right_on=["model_id", "processed"],
# )

# # optional: drop the mapping key col from the right side
# out = out.drop(columns=["processed"])

# # 5) report match rate
# missing = out["original"].isna().sum()
# print(f"Rows: {len(out)} | Missing originals: {missing}")

# # 6) save
# out.to_csv(OUT_CSV, index=False)
# print(f"Wrote: {OUT_CSV}")