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
MAKE_RECOMMENDATION = True
specific_deneme_count = True
SPECIFIC_DENEME = ["8", "9", "19"] if specific_deneme_count else False
specific_iter = iter(SPECIFIC_DENEME) if SPECIFIC_DENEME else None
ground_truth ={}
non_null_count = 0
import os

feature_folder_q = "8-CRITERIA_SELECTION/user_intent/quality_features"
feature_folder_e = "8-CRITERIA_SELECTION/user_intent/essential_features"
feature_folder_p = "8-CRITERIA_SELECTION/user_intent/preference_features"

for paper_key, paper in rationale_input.items():
    if not paper:
        continue
    paper_id = paper_key.split("_")[1]   # "paper_86" -> "86"
    # BURASI FEATURE BUNDLEI OLANLARI SKIPLEMEK ICIN: MAKE_RECOOMENDATION TRUE IS KALDIR
    eval_filename = f"eval_A{paper_id}.json"
    eval_path_q = os.path.join(feature_folder_q, eval_filename)
    eval_path_e = os.path.join(feature_folder_e, eval_filename)
    eval_path_p = os.path.join(feature_folder_p, eval_filename)

    # ---- process paper here ----    
    paper_key = f"sample_{paper_id}"
    paper = rationale_input.get(paper_key)
    if not paper:
        print(f"Skipping {paper_key}: no data found.")
        continue  # paper is None or missing
    user_text = paper.get("user_intent")
    correct_model = paper.get("model_full_name")
    if user_text is None or correct_model is None:
        continue  # skip if either value is null
    non_null_count += 1
    
    DENEME = "A" + str(paper_id)

    if specific_deneme_count:
        if specific_iter:
            try:
                sd = next(specific_iter)
                DENEME = "A" + sd
            except StopIteration:
                break
        else:
            DENEME = "A" + str(paper_id)

    print(f"Processing {DENEME}")
    correct_model = rationale_input["sample_" + str(paper_id)]["model_full_name"]
    ground_truth[DENEME] = correct_model
    with open("8-CRITERIA_SELECTION/user_intent/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    GEMINI_API_KEY = "" 
    llm_client = LLMClient(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
        max_retries=5,
        retry_delay_seconds=20.0,
    )
    # wrap it with a logger
    Elogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir="8-CRITERIA_SELECTION/user_intent/essential_features",
        print_output=True,
        save_file= f"eval_{DENEME}.json",   # optional
    )
    Plogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir="8-CRITERIA_SELECTION/user_intent/preference_features",
        print_output=True,
        save_file= f"eval_{DENEME}.json",   # optional
    )
    Qlogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir="8-CRITERIA_SELECTION/user_intent/quality_features",
        print_output=True,
        save_file= f"eval_{DENEME}.json",   # optional
    )

    Eextractor = EssentialFeaturesExtractor(Elogger)
    Pextractor = PreferenceFeaturesExtractor(Plogger)
    Qextractor = QualityFeaturesExtractor(Qlogger)


    
    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)

    # ====================
    # HER SEFERINDE RUNLAMA
    # ====================
    if not MAKE_RECOMMENDATION:
        if not os.path.exists(eval_path_e):
            print(f"Running EssentialFeaturesExtractor for {paper_key}...")
            time.sleep(60)
            Efeatures = Eextractor.extract(user_text)
        if not os.path.exists(eval_path_p):
            print(f"Running PreferenceFeaturesExtractor for {paper_key}...")
            time.sleep(60)
            Pfeatures = Pextractor.extract(user_text)
        if not os.path.exists(eval_path_q):
            print(f"Running QualityFeaturesExtractor for {paper_key}...")
            time.sleep(60)
            Qfeatures = Qextractor.extract(user_text)

        continue

    # ====================
    # SAVEDEN FEATURELARIN YUKLEME (BUNU FONKSIYONA MI CEVIRSEN?)
    # ====================

    import json
    import glob
    from E_utils import parse_llm_json_flex 
    from EC_PreferenceFeatureExtractor import get_llm_text, to_categorical_feat, to_numeric_feat, to_bool_feat, to_recency_feat

    Efeatures_raw = glob.glob(f"8-CRITERIA_SELECTION/user_intent/essential_features/eval_{DENEME}.json")
    Pfeatures_raw = glob.glob(f"8-CRITERIA_SELECTION/user_intent/preference_features/eval_{DENEME}.json")
    Qfeatures_raw = glob.glob(f"8-CRITERIA_SELECTION/user_intent/quality_features/eval_{DENEME}.json")

    for fpath in Efeatures_raw:
        with open(fpath, "r") as f:
            raw = json.load(f)
        # the actual model output
        raw_text = get_llm_text(raw[0])  # raw[0] cunku birden fazla llm outputu cikma ihitmaline karsi liste olarak kaydediyoruz outputlari
        Efeatures_data = parse_llm_json_flex(raw_text)

    for fpath in Pfeatures_raw:
        with open(fpath, "r") as f:
            raw = json.load(f)
        # the actual model output
        raw_text = get_llm_text(raw[0])  # raw[0] cunku birden fazla llm outputu cikma ihitmaline karsi liste olarak kaydediyoruz outputlari
        Pfeatures_Data = parse_llm_json_flex(raw_text)

    for fpath in Qfeatures_raw:
        with open(fpath, "r") as f:
            raw = json.load(f)
        # the actual model output
        raw_text = get_llm_text(raw[0])  # raw[0] cunku birden fazla llm outputu cikma ihitmaline karsi liste olarak kaydediyoruz outputlari
        Qfeatures_data = parse_llm_json_flex(raw_text)

    Efeatures = EssentialFeatures(
                task=to_categorical_feat(Efeatures_data.get("task")),
                domain=to_categorical_feat(Efeatures_data.get("domain")),
                model_name=to_categorical_feat(Efeatures_data.get("model_name")),
                author=to_categorical_feat(Efeatures_data.get("author")),
                objective=to_categorical_feat(Efeatures_data.get("objective")),
                task_aliases=to_categorical_feat(Efeatures_data.get("task_aliases")),
                domain_aliases=to_categorical_feat(Efeatures_data.get("domain_aliases")),
            )
    Qfeatures = QualityFeatures(Functional_Suitability=Qfeatures_data.get("Functional_Suitability"),
                Compatibility=Qfeatures_data.get("Compatibility"),
                Performance_Efficiency=Qfeatures_data.get("Performance_Efficiency"),
                Reliability=Qfeatures_data.get("Reliability"),
                Interaction_Capability=Qfeatures_data.get("Interaction_Capability"),
                Security=Qfeatures_data.get("Security"),
                Maintainability=Qfeatures_data.get("Maintainability"),
                Flexibility=Qfeatures_data.get("Flexibility")
                )
    Pfeatures = PreferenceFeatures(
                basemodels=to_categorical_feat(Pfeatures_Data.get("basemodels")),
                license_name=to_categorical_feat(Pfeatures_Data.get("license_name")),
                downloads_all_time=to_numeric_feat(Pfeatures_Data.get("downloads_all_time")),
                downloads_last_30_days=to_numeric_feat(Pfeatures_Data.get("downloads_last_30_days")),
                file_count=to_numeric_feat(Pfeatures_Data.get("file_count")),
                gated=to_bool_feat(Pfeatures_Data.get("gated")),
                lastModified=to_recency_feat(Pfeatures_Data.get("lastModified")),
                library_name=to_categorical_feat(Pfeatures_Data.get("library_name")),
                likes=to_numeric_feat(Pfeatures_Data.get("likes")),
                tensors_total=to_numeric_feat(Pfeatures_Data.get("tensors_total")),
                usedStorage=to_numeric_feat(Pfeatures_Data.get("usedStorage")),
                datasets=to_categorical_feat(Pfeatures_Data.get("datasets")),
                language=to_categorical_feat(Pfeatures_Data.get("language")),
                metrics=to_categorical_feat(Pfeatures_Data.get("metrics")),
            )

    # === CREATING FEATURE BUNDLE === 
    print("creating feature bundle")
    Fbundle = FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,  
    )
    # print(Fbundle.preferences.license_name.include)

    # === QUERY BUILDER === 
    # from EE_Query_Builder import ESQueryBuilder
    print("creating elastic search query")

    from elasticsearch import Elasticsearch, helpers
    from EE_Query_Builder_Advanced import ESQueryBuilder
    from EE_Query_Builder_Advanced import query_mapping

    from EE_Query_Builder_Adaptive import ESQueryBuilderAdaptive
    from EE_Query_Builder_Advanced import ESQueryBuilder


    ES_URL = "http://localhost:9200"
    INDEX_NAME = "models_t7"

    es = Elasticsearch(ES_URL)
    # # ORIGINAL RUNNING METHOD FOR QUERY BUILDER
    # builder = ESQueryBuilder(
    #     mapping=query_mapping,
    #     target_hits=20,   # how many results you want before stopping
    #     size=10
    # )

    # stages = builder.build_plan(
    #     features=Fbundle,
    # )

    # response, final_query = builder.search(
    #     es_client=es,
    #     index=INDEX_NAME,
    #     stages=stages
    # )
    # ADAPTIVE
    builder = ESQueryBuilderAdaptive(
    mapping=query_mapping,
    target_hits=150,
    size=100
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
            # Support both raw and reshaped hit structures
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

    from E_utils import save_to_json, object_to_dict
    q_path = "8-CRITERIA_SELECTION/user_intent/query_output/eval_"
    r_path = "8-CRITERIA_SELECTION/user_intent/recommendation_output/eval_"
    b_path = "8-CRITERIA_SELECTION/user_intent/feature_bundle/eval_"
    s_path = "8-CRITERIA_SELECTION/user_intent/stages_output/eval_"
    f_path = "8-CRITERIA_SELECTION/user_intent/functional_features/eval_"

    save_to_json(Ffeatures, f_path, DENEME)
    DENEME = "M" + str(paper_id)
    save_to_json(final_query, q_path, DENEME)
    save_to_json(response, r_path, DENEME)
    Fbundle_dict = object_to_dict(Fbundle)
    save_to_json(Fbundle_dict, b_path, DENEME)
    # save_to_json(stages, s_path, DENEME)
    save_to_json(final_feature_groups, s_path, DENEME)

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


