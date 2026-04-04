import json
with open("11-RECOMMENDATION_EVALUATION/paper_model_2/hf_case_study_candidates_openalex_3F.json", "r", encoding="utf-8") as f:
    data = json.load(f)

    print(len(data))