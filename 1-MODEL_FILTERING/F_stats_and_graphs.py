import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# SON HALI F_PLOTS_MODELS

with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f: #SU ZIKKIMI TAMAMLA ONCE
    stats = json.load(f)

# Assume your list of dicts is called 'models'
df = pd.DataFrame(stats)

####====##### Plot most popular library_name and pipeline_tag (top 15)

# Top 15 library names
top_libraries = df['library_name'].value_counts().head(15)
top_libraries.plot(kind='bar', figsize=(12,6), title='Top 15 Library Names')
plt.xticks(rotation=45)  # tilt x-axis labels
plt.ylabel('Count')
plt.show()

# Top 15 pipeline tags
top_pipelines = df['pipeline_tag'].value_counts().head(15)
ax = top_pipelines.plot(kind='bar', figsize=(12,6), title='Top 15 Pipeline Tags')
plt.ylabel('Count')
plt.xticks(rotation=45)  # tilt x-axis labels
plt.tight_layout()       # optional, avoids cutting off labels
plt.show()


####====##### Numeric keys: downloads_last_30_days, downloads_all_time, likes

numeric_keys = ['downloads_last_30_days', 'downloads_all_time', 'likes']

counts = {key: df[key].notnull().sum() for key in numeric_keys}
pd.Series(counts).plot(kind='bar', figsize=(10,5), title='Number of Items with Each Numeric Value')
plt.xticks(rotation=45)  # tilt x-axis labels
plt.ylabel('Number of Items')
plt.show()


####====##### Boolean keys: private, gated, disabled
bool_keys = ['private', 'gated', 'disabled']

bool_counts = {key: df[key].value_counts() for key in bool_keys}

# Prepare DataFrame for plotting
bool_df = pd.DataFrame(bool_counts).fillna(0)  # fill missing True/False with 0

bool_df.T.plot(kind='bar', figsize=(10,6), title='Boolean Features: True/False Counts')
plt.ylabel('Count')
plt.show()

####====##### usedStorage rounding and counting
# Round to nearest 1e8
df['usedStorageRounded'] = (df['usedStorage'] / 1e8).round() * 1e8

used_storage_counts = df['usedStorageRounded'].value_counts().sort_index()

used_storage_counts.plot(kind='bar', figsize=(12,6), title='Used Storage Rounded to Closest 1e8')
plt.ylabel('Number of Items')
plt.show()


####====#####  plot for all keys with nulls
# DataFrame indicating nulls
null_counts = df.isnull().sum()  # number of nulls per column
not_null_counts = df.notnull().sum()  # number of non-null per column

# Combine into one DataFrame
null_summary = pd.DataFrame({
    'has_value': not_null_counts,
    'is_null': null_counts
})

print(null_summary)

# Filter only columns that have any nulls
null_cols = null_summary[null_summary['is_null'] > 0]

null_cols.plot(kind='bar', figsize=(12,6), title='Non-null vs Null Counts per Key')
plt.ylabel('Number of Items')
plt.show()


####====##### 

####====##### 
