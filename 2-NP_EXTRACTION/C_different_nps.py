#BURANIN AMACI YENI FILTRELENMIS MODELLERLE ESKISI ARASINDAKI NP FARKINI BULMAK

import json

with open("2-NP_EXTRACTION/NP_global_dictionary_suan_filtered_2.json", "r") as f:
    suan = json.load(f)


suan_np_list = []

for np_dic in suan.values():
    np = np_dic["noun_phrase"]
    suan_np_list.append(np)

long_suan_np_list = []
for np in suan_np_list:
    if len(np) > 50:
        long_suan_np_list.append(np)

output_path = "2-NP_EXTRACTION/all_nps.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(suan_np_list, f, indent=2, ensure_ascii=False)