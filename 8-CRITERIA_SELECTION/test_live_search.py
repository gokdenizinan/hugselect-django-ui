import os
import sys
from pathlib import Path
from pprint import pprint

from elasticsearch import Elasticsearch

from EA_Features import FeatureBundle, FunctionalFeatures, QualityFeatures
from EB_LLM_Client import LLMClient, LoggingLLMClient

from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor
from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor
from EC_QualityFeatureExtractor import QualityFeaturesExtractor
from EC_FunctionalFeatureExtractor import NounPhraseExtractor

from EE_Query_Builder_Clean_modified_v3_dedupfix import (
    ESQueryBuilderAdaptive,
    query_mapping,
)

try:
    from E_utils import object_to_dict
except ImportError:
    object_to_dict = None


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs" / "live_search"

ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"

ENABLE_RANK_FUNCTIONS = True
ENABLE_QUALITY_DIMENSIONS = False
ENABLE_FEATURE_LOCATIONS = False

FEATURE_WEIGHT_GROUPS = {
    "essential": {
        "task": 11.5,
        "domain": 10.5,
        "author": 2.5,
        "objective": 10.0,
        "model_name": 8.0,
    },
    "preference": {
        "license_name": 8.0,
        "library_name": 1.8,
        "basemodels": 10.8,
        "datasets": 1.8,
        "language": 9.5,
        "metrics": 1.0,
    },
    "functional": {
        "functional_item": 12.0,
    },
    "quality": {
        "Functional_Suitability": 2.0,
        "Compatibility": 1.2,
        "Performance_Efficiency": 1.2,
        "Reliability": 1.2,
        "Interaction_Capability": 1.0,
        "Security": 1.0,
        "Maintainability": 1.0,
        "Flexibility": 1.2,
    },
    "rank": {
        "likes": 0.8,
        "downloads_last_30_days": 0.75,
        "Functional_Suitability": 1.2,
        "Compatibility": 0.8,
        "Performance_Efficiency": 0.8,
        "Reliability": 0.9,
        "Interaction_Capability": 0.7,
        "Security": 0.8,
        "Maintainability": 0.7,
        "Flexibility": 0.8,
    },
}

BOOST_CONFIG = {
    "rank": {
        "max": 70.0,
    },
    "match_mode": {
        "grams_factor": 0.90,
    },
}

PRIORITY_MULTIPLIERS = {
    "must": 1.8,
    "strong_prefer": 1.4,
    "prefer": 1.0,
    "avoid": 0.0,
}


def show_object(title, obj):
    print(f"\n--- {title} ---")
    if object_to_dict is not None:
        try:
            pprint(object_to_dict(obj), width=120)
            return
        except Exception:
            pass
    pprint(obj, width=120)


def extract_feature_bundle(user_text: str) -> FeatureBundle:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Run:\n"
            "export GEMINI_API_KEY='your-key-here'\n"
            "Do not paste your real key into ChatGPT."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(
        api_key=api_key,
        model_name="gemini-2.5-flash",
        max_retries=2,
        retry_delay_seconds=3.0,
    )

    Elogger = LoggingLLMClient(llm_client, save_dir=str(LOG_DIR), print_output=False, save_file="essential.json")
    Plogger = LoggingLLMClient(llm_client, save_dir=str(LOG_DIR), print_output=False, save_file="preference.json")
    Qlogger = LoggingLLMClient(llm_client, save_dir=str(LOG_DIR), print_output=False, save_file="quality.json")

    Efeatures = EssentialFeaturesExtractor(Elogger).extract(user_text)
    Pfeatures = PreferenceFeaturesExtractor(Plogger).extract(user_text)
    try:
        Qfeatures = QualityFeaturesExtractor(Qlogger).extract(user_text)
    except Exception as e:
        print("\nWARNING: Quality extraction failed. Continuing with empty QualityFeatures.")
        print(f"Quality extraction error: {type(e).__name__}: {e}")
        Qfeatures = QualityFeatures()

    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(user_text, Fextractor)

    return FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )


def make_builder() -> ESQueryBuilderAdaptive:
    return ESQueryBuilderAdaptive(
        mapping=query_mapping,
        target_hits=50,
        size=10,
        feature_weight_groups=FEATURE_WEIGHT_GROUPS,
        boost_config=BOOST_CONFIG,
        enable_rank_functions=ENABLE_RANK_FUNCTIONS,
        enable_quality_dimensions=ENABLE_QUALITY_DIMENSIONS,
        enable_feature_locations=ENABLE_FEATURE_LOCATIONS,
        priority_multipliers=PRIORITY_MULTIPLIERS,
        synonym_min_conf=0.50,
        minimum_should_match=1,
        synonym_cache_path=str(BASE_DIR / "synonym_cache.json"),
    )


def print_hits(response):
    hits = response.get("hits", {}).get("hits", [])

    print(f"\nTOP {len(hits)} RESULTS:")
    for i, hit in enumerate(hits, start=1):
        src = hit.get("_source", {}) or {}

        model_id = (
            hit.get("pretty_id")
            or src.get("modelID")
            or src.get("model_id")
            or hit.get("_id")
        )

        metadata = src.get("Metadata", {}) or {}

        print(f"\n{i}. {model_id}")
        print(f"   ES score: {hit.get('_score')}")
        print(f"   Display score: {hit.get('display_score')}")
        print(f"   Task: {metadata.get('pipeline_tag')}")
        print(f"   Library: {metadata.get('library_name')}")
        print(f"   License: {metadata.get('license')}")
        print(f"   Likes: {metadata.get('likes')}")
        print(f"   Downloads 30d: {metadata.get('downloads_last_30_days')}")


def main():
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = (
            "I need a text classification model for Turkish sentiment analysis, "
            "preferably based on BERT and usable commercially."
        )

    print("\nUSER TEXT:")
    print(user_text)

    print("\n[1] Extracting FeatureBundle...")
    bundle = extract_feature_bundle(user_text)
    show_object("FeatureBundle", bundle)

    print("\n[2] Connecting to Elasticsearch...")
    es_client = Elasticsearch(ES_URL)

    if not es_client.indices.exists(index=INDEX_NAME):
        raise RuntimeError(f"Elasticsearch index does not exist: {INDEX_NAME}")

    print(f"Connected. Index exists: {INDEX_NAME}")

    print("\n[3] Building query and searching...")
    builder = make_builder()
    prebuilt_groups = builder.precompute_feature_group_cache(bundle)

    response, final_query, final_feature_groups = builder.search(
        es_client=es_client,
        index=INDEX_NAME,
        features=bundle,
        prebuilt_groups=prebuilt_groups,
        include_explain=False,
    )

    show_object("FeatureGroups", final_feature_groups)
    print_hits(response)

    print("\nDONE: live extraction + Elasticsearch search worked.")


if __name__ == "__main__":
    main()
