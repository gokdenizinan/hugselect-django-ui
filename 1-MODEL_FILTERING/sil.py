import json

with open("1-MODEL_FILTERING/models_2025.json", "r", encoding = "utf-8") as f:
    models_2025 = json.load(f)

with open("1-MODEL_FILTERING/hf_model_download_stats.json", "r", encoding = "utf-8") as f:
    download_dic = json.load(f)

download_list = [list(d.keys())[0] for d in download_dic]

olanlar = []
for i in models_2025:
    if i in download_list:
        olanlar.append(i)
print(len(olanlar))