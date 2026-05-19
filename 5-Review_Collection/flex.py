import json

with open("5-REVIEW_COLLECTION/stackoverglow_mentioned.json", "r", encoding = "utf-8") as f:
    stack = json.load(f)

names = []
for dic in stack:
    m_name = dic["searched_topic"]
    names.append(m_name)

print(names)
print(len(names))
