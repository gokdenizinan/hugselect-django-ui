import re
import pandas as pd
df = pd.read_csv("4-LLM_FEATURE_ORGANIZATION/df_feature_info_global.csv")
repeated_features = []

pattern = re.compile(r"^\s*(.*?)\s*\(\s*(.*?)\s*\)\s*$")

for value in df["Feature"].dropna():
    match = pattern.match(value)
    if match:
        outside, inside = match.groups()
        if outside == inside:
            repeated_features.append(outside)

import json

with open("4-LLM_FEATURE_ORGANIZATION/repeated_features.json", "w") as f:
    json.dump(repeated_features, f, indent=2)
