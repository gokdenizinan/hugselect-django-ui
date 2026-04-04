import os
import json
import re

PHASE = 0

if PHASE == 0: 
    def count_sentences(text):
        """Count sentences in a text string using regex."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]
        return len(sentences)

    def get_counts(folder_path, old = False):
        # Initialize counters
        sentence_counts = {"reddit": 0, "stack": 0, "hf": 0}
        nonempty_file_counts = {"reddit": 0, "stack": 0, "hf": 0}
        nonempty_file_names = {"reddit": [], "stack": [], "hf": [], "total" : set()}
        # Loop through all JSON files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Track whether each key is non-empty for this file
                for key in ["reddit", "stack", "hf"]:
                    if key in data and data[key]:  # exists and not empty
                        nonempty_file_counts[key] += 1  # count the file
                        new_filename = filename.replace("_", "/")
                        nonempty_file_names[key].append(new_filename.removesuffix(".json"))
                        nonempty_file_names["total"].add(new_filename.removesuffix(".json"))
                        for item in data[key]:
                            if not old:
                                for text in item.get("mentioned", []):
                                    sentence_counts[key] += count_sentences(text)
                            else: 
                                title = item.get("title", "")
                                body = item.get("body", "")
                                if title:
                                    sentence_counts[key] += 1
                                if body:
                                    sentence_counts[key] += 1
                                for text in item.get("comments", []):
                                    sentence_counts[key] += count_sentences(text)
        
        return sentence_counts, nonempty_file_counts, nonempty_file_names


    # # CONFIGURATION
    # print("---- ---- ---- ---- ----")
    # print("A")
    # folder_a = "5-REVIEW_COLLECTION/united_reviews_by_model"
    # files_a = [f for f in os.listdir(folder_a) if f.endswith(".json")]
    # print(f"📂 Folder A ({folder_a}) contains {len(files_a)} files.")
    # sentence_counts, nonempty_file_counts, nonempty_file_names = get_counts(folder_a, old = True)
    # print("Total sentence counts:", sentence_counts)
    # print("Files with non-empty data:", nonempty_file_counts)
    # print("total model count:", len(nonempty_file_names["total"]))

    print("---- ---- ---- ---- ----")
    print("B")

    folder_b = "5-REVIEW_COLLECTION/united_f7_not_in_n_dict"
    files_b = [f for f in os.listdir(folder_b) if f.endswith(".json")]
    print(f"📂 Folder B ({folder_b}) contains {len(files_b)} files.")
    sentence_counts, nonempty_file_counts , nonempty_file_names= get_counts(folder_b)
    print("Total sentence counts:", sentence_counts)
    print("Files with non-empty data:", nonempty_file_counts)
    print("total model count:", len(nonempty_file_names["total"]))


    print("---- ---- ---- ---- ----")
    print("C")

    folder_c = "5-REVIEW_COLLECTION/united_f3"
    files_c = [f for f in os.listdir(folder_c) if f.endswith(".json")]
    print(f"📂 Folder C ({folder_c}) contains {len(files_c)} files.")
    sentence_counts, nonempty_file_counts, nonempty_file_names = get_counts(folder_c)
    print("Total sentence counts:", sentence_counts)
    print("Files with non-empty data:", nonempty_file_counts)
    print("total model count:", len(nonempty_file_names["total"]))

    print("---- ---- ---- ---- ----")
    print("D")
    folder_d = "5-REVIEW_COLLECTION/united_f5"
    files_d = [f for f in os.listdir(folder_d) if f.endswith(".json")]
    print(f"📂 Folder C ({folder_d}) contains {len(files_d)} files.")
    sentence_counts, nonempty_file_counts, nonempty_file_names = get_counts(folder_d)
    print("Total sentence counts:", sentence_counts)
    print("Files with non-empty data:", nonempty_file_counts)
    print("total model count:", len(nonempty_file_names["total"]))

# en populer 1000 modelle karsilastir
PHASE = 3
if PHASE == 2:

    import os
    import json
    import nltk

    # If you haven’t downloaded NLTK sentence tokenizer yet:
    # nltk.download("punkt")

    FOLDER_PATH = "5-REVIEW_COLLECTION/united_f5"  # 🔁 change this to your folder path

    def count_sentences(text_list):
        """Count total sentences in a list of text strings."""
        if not text_list:
            return 0
        return sum(len(nltk.sent_tokenize(text)) for text in text_list)

    # Store model_id → total_sentence_count
    model_sentence_counts = {}

    # Loop through all JSON files in the folder
    for filename in os.listdir(FOLDER_PATH):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(FOLDER_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_id = data.get("model_id", filename)
        stack_items = data.get("stack", [])

        total_sentences = 0
        for item in stack_items:
            mentioned_list = item.get("comments", []) #mentioned yap untef_f ler icin
            total_sentences += count_sentences(mentioned_list)

        model_sentence_counts[model_id] = total_sentences

    # Sort by total sentence count (descending)
    sorted_models = sorted(model_sentence_counts.items(), key=lambda x: x[1], reverse=True)

    # Print ordered list
    print("Model_ID | Sentence_Count")
    print("-" * 40)
    for model_id, count in sorted_models:
        print(f"{model_id:20} {count}")

    # (Optional) Save results as JSON for analysis
    output_path = os.path.join("5-REVIEW_COLLECTION", "sentence_count_summary_Initial.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_models, f, ensure_ascii=False, indent=2)

    print(f"\nSaved ordered sentence counts to: {output_path}")

# en populer 1000 modelle karsilastir

PHASE = 0
if PHASE == 1:
    with open("1-MODEL_FILTERING/N_sorted_model_likes_P9.json", "r", encoding="utf-8") as f:
        p9 = json.load(f)
    
    with open("1-MODEL_FILTERING/N_sorted_model_likes_Y1.json", "r", encoding="utf-8") as f:
        y1 = json.load(f)

    p9_keys = list(p9.keys())[:100]
    y1_keys = list(y1.keys())[:100]
    
    y1_p9 = []
    for model in y1_keys:
        if model not in p9_keys:
            y1_p9.append(model)

    print(len(y1_p9))
    print(y1_p9)

    print(p9_keys[:10])
    print(y1_keys[:10])
    reddit = nonempty_file_names["reddit"]
    hf = nonempty_file_names["hf"]
    stack = nonempty_file_names["stack"]
    

    # REDDIT
    y1_f3 = []
    for model in y1_keys:
        if model not in reddit:
            y1_f3.append(model)
    print("reddit")
    print(reddit[:10])
    print(len(y1_f3))

    p9_f3 = []
    for model in p9_keys:
        if model not in reddit:
            p9_f3.append(model)

    print(len(p9_f3))

    # STACK
    y1_f3 = []
    for model in y1_keys:
        if model not in stack:
            y1_f3.append(model)
    print("stack")
    print(stack[:10])

    print(len(y1_f3))

    p9_f3 = []
    for model in p9_keys:
        if model not in stack:
            p9_f3.append(model)

    print(len(p9_f3))

    # HF

    y1_f3 = []
    for model in y1_keys:
        if model not in hf:
            y1_f3.append(model)
    print("hf")
    print(hf[:10])
    print(len(y1_f3))

    p9_f3 = []
    for model in p9_keys:
        if model not in hf:
            p9_f3.append(model)

    print(len(p9_f3))

        # TOTAL

    total = set()

    for a in stack:
        total.add(a)
    for a in hf:
        total.add(a)
    for a in reddit:
        total.add(a)

    y1_f3 = []
    for model in y1_keys:
        if model not in total:
            y1_f3.append(model)
    print("total")
    print("LEN TOTAL", len(total))
    print(len(y1_f3))

    p9_f3 = []
    for model in p9_keys:
        if model not in total:
            p9_f3.append(model)

    print(len(p9_f3))
    print(p9_f3)