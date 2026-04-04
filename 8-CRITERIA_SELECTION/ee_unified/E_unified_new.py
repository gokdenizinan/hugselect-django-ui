from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
import time
import json

print("starting....")

# === FEATURES ===
print("importing feature classes...")
from EA_Features import (
    EssentialFeatures,
    PreferenceFeatures,
    QualityFeatures,
    FeatureBundle,
    UserQuery,
    ModelResult
)

# === LLM CLIENT ===
print("importing llm client...")
from EB_LLM_Client import LLMClient, LoggingLLMClient

# === FEATURE EXTRACTOR ===
print("importing ALL feature extractors...")

print("importing functional feature extractor...")

from EC_FunctionalFeatureExtractor import NounPhraseExtractor
from EC_FunctionalFeatureExtractor import FunctionalFeatures

print("importing essential feature extractor...")

from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor

print("importing preference feature extractor...")

from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor

print("importing quality feature extractor...")

from EC_QualityFeatureExtractor import QualityFeaturesExtractor

print("importing user input...")

from ED_User_Input import user_inputs, rationale_input

print("running ALL feature extractors...")

# ============
# BUNU SETLE
# ============
import os
import json
import glob
import time

from E_utils import parse_llm_json_flex, save_to_json, object_to_dict
from EC_PreferenceFeatureExtractor import (get_llm_text,to_categorical_feat,to_numeric_feat,to_bool_feat,to_recency_feat,)

from elasticsearch import Elasticsearch

# from EE_Query_Builder_Adaptive import query_mapping
# from EE_Query_Builder_Adaptive import ESQueryBuilderAdaptive

from EE_Query_Builder_All import query_mapping
from EE_Query_Builder_All import ESQueryBuilderAdaptive

# A FUNCTION
def load_single_feature_json(folder_path, eval_id):
    """
    Loads a single saved eval file like eval_A8.json and returns parsed JSON content.
    """
    pattern = os.path.join(folder_path, f"eval_{eval_id}.json")
    matches = glob.glob(pattern)

    if not matches:
        print(f"File not found: {pattern}")
        return None

    fpath = matches[0]
    with open(fpath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print(f"Empty JSON content in: {fpath}")
        return None

    raw_text = get_llm_text(raw[0])
    return parse_llm_json_flex(raw_text)

# ====================
# CONFIG
# ====================
# 69 u da runla sonra birak
MAKE_RECOMMENDATION = True
specific_deneme_count = True
OUTPUT_LETTER = "M"
GEMINI_API_KEY = "REMOVED_GEMINI_API_KEY"

SPECIFIC_DENEME = {"19"} if specific_deneme_count else None

feature_folder_q = "8-CRITERIA_SELECTION/user_intent/quality_features"
feature_folder_e = "8-CRITERIA_SELECTION/user_intent/essential_features"
feature_folder_p = "8-CRITERIA_SELECTION/user_intent/preference_features"

ground_truth = {}
non_null_count = 0

ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"

for original_key, original_paper in rationale_input.items():
    if not original_paper:
        continue

    parts = original_key.split("_")
    if len(parts) < 2:
        print(f"Skipping malformed key: {original_key}")
        continue

    paper_id = parts[1]

    # Only process selected IDs if specific filtering is enabled
    if SPECIFIC_DENEME is not None and paper_id not in SPECIFIC_DENEME:
        continue

    sample_key = f"sample_{paper_id}"
    paper = rationale_input.get(sample_key)

    if not paper:
        print(f"Skipping {sample_key}: no data found.")
        continue

    user_text = paper.get("user_intent")
    correct_model = paper.get("model_full_name")

    if user_text is None or correct_model is None:
        print(f"Skipping {sample_key}: missing user_intent or model_full_name.")
        continue

    non_null_count += 1

    eval_id = f"A{paper_id}"
    output_id = f"{OUTPUT_LETTER}{paper_id}"

    print(f"Processing {eval_id}")

    # Save ground truth
    ground_truth[eval_id] = correct_model
    with open("8-CRITERIA_SELECTION/user_intent/ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    eval_path_q = os.path.join(feature_folder_q, f"eval_{eval_id}.json")
    eval_path_e = os.path.join(feature_folder_e, f"eval_{eval_id}.json")
    eval_path_p = os.path.join(feature_folder_p, f"eval_{eval_id}.json")

    # ====================
    # LLM CLIENTS
    # ====================
    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )

    Elogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_e,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )
    Plogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_p,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )
    Qlogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=feature_folder_q,
        print_output=True,
        save_file=f"eval_{eval_id}.json",
    )

    Eextractor = EssentialFeaturesExtractor(Elogger)
    Pextractor = PreferenceFeaturesExtractor(Plogger)
    Qextractor = QualityFeaturesExtractor(Qlogger)

    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)

    # ====================
    # FEATURE EXTRACTION MODE
    # ====================
    if not MAKE_RECOMMENDATION:
        if not os.path.exists(eval_path_e):
            print(f"Running EssentialFeaturesExtractor for {sample_key}...")
            time.sleep(60)
            Eextractor.extract(user_text)
        else:
            print(f"Skipping essential extraction, already exists: {eval_path_e}")

        if not os.path.exists(eval_path_p):
            print(f"Running PreferenceFeaturesExtractor for {sample_key}...")
            time.sleep(60)
            Pextractor.extract(user_text)
        else:
            print(f"Skipping preference extraction, already exists: {eval_path_p}")

        if not os.path.exists(eval_path_q):
            print(f"Running QualityFeaturesExtractor for {sample_key}...")
            time.sleep(60)
            Qextractor.extract(user_text)
        else:
            print(f"Skipping quality extraction, already exists: {eval_path_q}")

        continue

    # ====================
    # LOAD SAVED FEATURES
    # ====================
    Efeatures_data = load_single_feature_json(feature_folder_e, eval_id)
    Pfeatures_data = load_single_feature_json(feature_folder_p, eval_id)
    Qfeatures_data = load_single_feature_json(feature_folder_q, eval_id)

    if Efeatures_data is None or Pfeatures_data is None or Qfeatures_data is None:
        print(f"Skipping {eval_id}: one or more feature files missing or invalid.")
        continue

    # ====================
    # CONSTRUCT FEATURE OBJECTS
    # ====================
    Efeatures = EssentialFeatures(
        task=to_categorical_feat(Efeatures_data.get("task")),
        domain=to_categorical_feat(Efeatures_data.get("domain")),
        model_name=to_categorical_feat(Efeatures_data.get("model_name")),
        author=to_categorical_feat(Efeatures_data.get("author")),
        objective=to_categorical_feat(Efeatures_data.get("objective")),
        task_aliases=to_categorical_feat(Efeatures_data.get("task_aliases")),
        domain_aliases=to_categorical_feat(Efeatures_data.get("domain_aliases")),
    )

    Qfeatures = QualityFeatures(
        Functional_Suitability=Qfeatures_data.get("Functional_Suitability"),
        Compatibility=Qfeatures_data.get("Compatibility"),
        Performance_Efficiency=Qfeatures_data.get("Performance_Efficiency"),
        Reliability=Qfeatures_data.get("Reliability"),
        Interaction_Capability=Qfeatures_data.get("Interaction_Capability"),
        Security=Qfeatures_data.get("Security"),
        Maintainability=Qfeatures_data.get("Maintainability"),
        Flexibility=Qfeatures_data.get("Flexibility"),
    )

    Pfeatures = PreferenceFeatures(
        basemodels=to_categorical_feat(Pfeatures_data.get("basemodels")),
        license_name=to_categorical_feat(Pfeatures_data.get("license_name")),
        downloads_all_time=to_numeric_feat(Pfeatures_data.get("downloads_all_time")),
        downloads_last_30_days=to_numeric_feat(Pfeatures_data.get("downloads_last_30_days")),
        file_count=to_numeric_feat(Pfeatures_data.get("file_count")),
        gated=to_bool_feat(Pfeatures_data.get("gated")),
        lastModified=to_recency_feat(Pfeatures_data.get("lastModified")),
        library_name=to_categorical_feat(Pfeatures_data.get("library_name")),
        likes=to_numeric_feat(Pfeatures_data.get("likes")),
        tensors_total=to_numeric_feat(Pfeatures_data.get("tensors_total")),
        usedStorage=to_numeric_feat(Pfeatures_data.get("usedStorage")),
        datasets=to_categorical_feat(Pfeatures_data.get("datasets")),
        language=to_categorical_feat(Pfeatures_data.get("language")),
        metrics=to_categorical_feat(Pfeatures_data.get("metrics")),
    )

    # ====================
    # CREATE FEATURE BUNDLE
    # ====================
    print("Creating feature bundle")
    Fbundle = FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )

    # ====================
    # ELASTIC QUERY
    # ====================
    print("Creating elastic search query")

    es = Elasticsearch(ES_URL)

    builder = ESQueryBuilderAdaptive(
        mapping=query_mapping,
        target_hits=150,
        size=100,
    )

    response, final_query, final_feature_groups = builder.search(
        es_client=es,
        index=INDEX_NAME,
        features=Fbundle,
        include_score_breakdown=True,
        include_explain=True,
    )

    def print_results(res):
        print("\nModels Matching Query:")
        hits_obj = res.get("hits", {}).get("hits", [])

        for hit in hits_obj:
            print("====  ====  ====  ====  ====. ====  ====  ====  ====  ====")

            score = hit.get("_score", hit.get("score"))
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "None"

            src = hit.get("_source", hit.get("source", {})) or {}

            model_id = src.get("modelID")
            tags = src.get("Metadata", {}).get("tags")
            features = src.get("Features")
            pipeline_tag = src.get("Metadata", {}).get("pipeline_tag")

            print(
                f"- score={score_str}, \n"
                f"modelID={model_id!r}, \n"
                f"=    =    =    =    =    =     = \n"
                f"pipeline_tag={pipeline_tag!r}, \n"
                f"tags={tags!r}, \n"
                f"Features={features!r} \n"
            )

    print("=====================================")
    print("=====================================")
    print_results(response)

    # ====================
    # SAVE OUTPUTS
    # ====================
    q_path = "8-CRITERIA_SELECTION/user_intent/query_output/eval_"
    r_path = "8-CRITERIA_SELECTION/user_intent/recommendation_output/eval_"
    b_path = "8-CRITERIA_SELECTION/user_intent/feature_bundle/eval_"
    s_path = "8-CRITERIA_SELECTION/user_intent/stages_output/eval_"
    f_path = "8-CRITERIA_SELECTION/user_intent/functional_features/eval_"

    save_to_json(Ffeatures, f_path, eval_id)
    save_to_json(final_query, q_path, output_id)
    save_to_json(response, r_path, output_id)

    Fbundle_dict = object_to_dict(Fbundle)
    save_to_json(Fbundle_dict, b_path, output_id)
    save_to_json(final_feature_groups, s_path, output_id)

print(f"NUMBER OF TESTED INPUTS: {non_null_count}")


# # ESKI BIR DENEMELIK RUN
# def example_query(es: Elasticsearch, index_name: str, body) -> None:

#     res = es.search(index=index_name, body=body)

#     print("\nModels Matching Query:")
#     for hit in res["hits"]["hits"]:
#         src = hit["_source"]
#         model_id = src.get("modelID")
#         tags = src.get("tags")
#         features = src.get("Features")
#         pipeline_tag = src.get("Metadata", {}).get("pipeline_tag")
#         print(f"- modelID={model_id!r}, tags={tags!r}, Features={features!r}, pipeline_tag={pipeline_tag!r}")

# # # RUN
# ES_URL = "http://localhost:9200"  # change if needed
# INDEX_NAME = "models_02"             # name of the ES index
# MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T9.json"
# DATA_FOLDER = "HF-Models-T9"

# es = Elasticsearch(ES_URL)
# ESCLIENT = ESQueryBuilder(query_mapping, DEFAULT_WEIGHTS)
# QUERY = ESCLIENT.build(Fbundle)
# print("")
# print("=====================================")
# print("Generated ES Query:")
# print(QUERY)
# print("")
# print("=====================================")
# example_query(es, INDEX_NAME, QUERY)
# print("=====================================")


