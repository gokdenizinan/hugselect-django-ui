import ijson
import json

# 1️⃣ Load the model dict once and prepare the sets
with open("5-REVIEW_COLLECTION/model_dict.json", "r", encoding="utf-8") as f:
    model_dict = json.load(f)

all_models = [entry["model_name"] for entry in model_dict.values()]
line = set(all_models[4000:33000])
line_2 = set(all_models[:4000] + all_models[33000:])

# 2️⃣ Open output files
split_1_file = open("5-REVIEW_COLLECTION/llm_check_reviews/split_1.json", "w", encoding="utf-8")
split_2_file = open("5-REVIEW_COLLECTION/llm_check_reviews/split_2.json", "w", encoding="utf-8")

split_1_file.write("[\n")
split_2_file.write("[\n")

first_1 = True
first_2 = True

# 3️⃣ Stream through the huge JSON file
with open("5-REVIEW_COLLECTION/united_reviews/united_reviews.json", "r", encoding="utf-8") as f:
    for item in ijson.items(f, "item"):
        topic = item["topic"]

        # Decide which split it goes to
        if topic in line:
            if not first_1:
                split_1_file.write(",\n")
            json.dump(item, split_1_file, ensure_ascii=False)
            first_1 = False
        elif topic in line_2:
            if not first_2:
                split_2_file.write(",\n")
            json.dump(item, split_2_file, ensure_ascii=False)
            first_2 = False
        # else: ignore if it doesn't match either set

# 4️⃣ Close the JSON arrays and files
split_1_file.write("\n]")
split_2_file.write("\n]")

split_1_file.close()
split_2_file.close()

print("Splitting complete!")
