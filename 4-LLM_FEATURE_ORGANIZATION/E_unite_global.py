import pandas as pd

df1 = pd.read_csv("4_LLM_FEATURE_ORGANIZATION/df_feature_info.csv")
df2 = pd.read_csv("4_LLM_FEATURE_ORGANIZATION/df_feature_info_global.csv")

df = pd.concat([df1, df2], join="inner", ignore_index=True)
df.to_csv("4_LLM_FEATURE_ORGANIZATION/df_feature_info_united.csv", index=False, encoding="utf-8")

df1 = pd.read_csv("4_LLM_FEATURE_ORGANIZATION/df_model_info.csv")
df2 = pd.read_csv("4_LLM_FEATURE_ORGANIZATION/df_model_info_global.csv")

df = pd.concat([df1, df2], join="inner", ignore_index=True)
df.to_csv("4_LLM_FEATURE_ORGANIZATION/df_model_info_united.csv", index=False, encoding="utf-8")

# FINAL ADJUSTMENTS
df_model = pd.read_csv("4_LLM_FEATURE_ORGANIZATION/df_model_info_united.csv")
