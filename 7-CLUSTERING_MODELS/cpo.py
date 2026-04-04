import pandas as pd

df = pd.read_csv("7-CLUSTERING_MODELS/model_classification_improved_more.csv")
print(df.columns)
for col in df.columns[:5]:
    print(f"Column: {col}")
    print(df[col].head(10))
    print("-" * 40)