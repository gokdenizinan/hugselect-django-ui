import json
import pandas as pd

# df_qu = pd.read_csv("/Users/ardacanseradali/Documents/Thesis_master/6-REVIEW_SENTIMENT_ANALYSIS/llm_check_meaning/sentiment_for_f5_united_3_llm_go.csv")
# df_ff = pd.read_csv("/Users/ardacanseradali/Documents/Thesis_master/4-LLM_FEATURE_ORGANIZATION/df_model_info.csv")

# df_ff['model_id'] = df_ff['Author'] + '/' + df_ff['Model Name']

# df_qu_count = df_qu.groupby('model_id', as_index=False)['reviews'].count()

# # Rename the column for clarity
# df_qu_count.rename(columns={'reviews': 'num_reviews'}, inplace=True)

# merged_df = pd.merge(df_ff, df_qu_count, on='model_id', how='outer')

# df_selected = merged_df[['model_id', 'num_reviews', '#Features']]

# # 2️⃣ Rename the column '#Features' to something cleaner (for example: 'num_features')
# df_selected = df_selected.rename(columns={'#Features': 'num_features'})
# df_selected['num_reviews'] = df_selected['num_reviews'].astype('Int64')
# df_selected['num_features'] = df_selected['num_features'].astype('Int64')


# df_selected.to_csv('/Users/ardacanseradali/Documents/Thesis_master/6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/ff_qu_stats.csv', index=False)

df_selected = pd.read_csv("/Users/ardacanseradali/Documents/Thesis_master/6-REVIEW_SENTIMENT_ANALYSIS/llm_quality_mapping/ff_qu_stats.csv")

print("TOTAL ROWS: " , len(df_selected))

# Rows where num_reviews is not null
count_reviews = df_selected['num_reviews'].notna().sum()

# Rows where num_features is not null
count_features = df_selected['num_features'].notna().sum()

# Rows where both are not null
count_both = df_selected[df_selected['num_reviews'].notna() & df_selected['num_features'].notna()].shape[0]

count_ff_1 = (df_selected['num_features'] > 1).sum()


print(f"Rows with num_reviews: {count_reviews}")
print(f"Rows with num_features: {count_features}")
print(f"Rows with num_reviews >1: {count_ff_1}")
print(f"Rows with both: {count_both}")
print("--------------------")
# 1️⃣ Load your dictionary file
# (Assuming it's a JSON file like "my_dict.json")
with open("/Users/ardacanseradali/Documents/Thesis_master/1-MODEL_FILTERING/N_sorted_model_likes_P9.json", "r") as f:
    popular = json.load(f)

# 2️⃣ Get the list of keys
dict_keys = list(popular.keys())#[:30000]  # Adjust slicing as needed

# 3️⃣ Count how many keys exist in the model_id column of your DataFrame
matches = df_selected['model_id'].isin(dict_keys)

# 4️⃣ Get counts
num_matches = matches.sum()
num_total = len(dict_keys)

print(f"Keys found in df: {num_matches}")
print(f"Total keys in dictionary: {num_total}")
print(f"Fraction matched: {num_matches}/{num_total} = {num_matches / num_total:.2%}")
print("--------------------")

mask = (
    df_selected['model_id'].isin(dict_keys)
    & df_selected['num_reviews'].notna()
    & df_selected['num_features'].notna()
)

# 4️⃣ Count how many rows meet all three conditions
count_all = mask.sum()
total_keys = len(dict_keys)

print(f"✅ Models in dict AND with both values: {count_all}")
print(f"📊 Fraction: {count_all}/{total_keys} = {count_all / total_keys:.2%}")
print("--------------------")

mask = (
    df_selected['model_id'].isin(dict_keys)
    & df_selected['num_reviews'].notna()
)

# 4️⃣ Count how many rows meet all three conditions
count_all = mask.sum()
total_keys = len(dict_keys)

print(f"✅ Models in dict AND with reviews: {count_all}")
print(f"📊 Fraction: {count_all}/{total_keys} = {count_all / total_keys:.2%}")

print("--------------------")

mask = (
    df_selected['model_id'].isin(dict_keys)
    & df_selected['num_features'].notna()
)

# 4️⃣ Count how many rows meet all three conditions
count_all = mask.sum()
total_keys = len(dict_keys)

print(f"✅ Models in dict AND with features: {count_all}")
print(f"📊 Fraction: {count_all}/{total_keys} = {count_all / total_keys:.2%}")