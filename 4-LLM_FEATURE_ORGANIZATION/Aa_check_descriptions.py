import json

with open("3-LLM_FEATURE_EXTRACTION/Descriptions_Z_global_suan_U.json", "r", encoding="utf-8") as f:
    desc_dict = json.load(f)

print(len(desc_dict.items()))

for key in list(desc_dict.keys()):
    s = desc_dict[key]["output"] 
    s = s.strip()
    s = s.strip('`')
    if s.lower().startswith("json"):
        s = s[4:].strip()
    
    # 2️⃣ Parse JSON
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        print("Warning: failed to parse JSON:", s)
        print(key)
