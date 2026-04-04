import json 

with open("1-MODEL_FILTERING/hf_model_all_stats.json", "r", encoding="utf-8") as f:
    model_stats = json.load(f)

print(len(model_stats))

nu_score = {}
nu_set = set()
# Loop through each dictionary
for i, model in enumerate(model_stats):
    null_keys = [key for key, value in model.items() if value is None]
    for i in null_keys:
        if i not in nu_set:
            nu_score[i] =  1
            nu_set.add(i)
        else:
            nu_score[i] += 1

poss = model_stats[0].keys()
print(nu_score)
print("#######3")
for featur in poss:
    if featur not in nu_score.keys() :
        print(featur)
 
 # bus usan basicleri vercek tablodaki
