import json


# with open("2-NP_EXTRACTION/NP_aggregatedVFULL.json", "r", encoding="utf-8") as f:
#     temp_np2 = json.load(f)

# print(len(temp_np2))

# with open("2-NP_EXTRACTION/NP_aggregatedV5.json", "r", encoding="utf-8") as f:
#     temp_np = json.load(f)

# print(len(temp_np))


import os

folder_path = 'HF-Models-F5'
file_count = sum(
    1 for entry in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, entry))
)

print(f"Number of files: {file_count}")



# # Copy dic with a new name
# import json
# with open("2_Secondary_S/np_filtered.json", "r", encoding="utf-8") as f:
#     temp_filt = json.load(f)

# newdict = temp_filt.copy()
# with open("3_LLM_Prompt/np_filt_copy_w4.json", "w", encoding="utf-8") as f:
#     json.dump(newdict, f, indent=2)

# print("Ddictionary copied successfully to np_filt_copy.json")