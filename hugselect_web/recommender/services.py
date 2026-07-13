import os
import sys
from pathlib import Path

from elasticsearch import Elasticsearch


ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"

REPO_ROOT = Path(__file__).resolve().parents[2]
CRITERIA_DIR = REPO_ROOT / "8-CRITERIA_SELECTION"

if str(CRITERIA_DIR) not in sys.path:
    sys.path.insert(0, str(CRITERIA_DIR))


def clean_results(response, limit=10):
    hits = response.get("hits", {}).get("hits", [])
    results = []

    for hit in hits[:limit]:
        src = hit.get("_source", {}) or {}
        metadata = src.get("Metadata", {}) or {}

        model_id = (
            hit.get("pretty_id")
            or src.get("modelID")
            or src.get("model_id")
            or hit.get("_id")
        )

        results.append({
            "model_id": model_id,
            "author": src.get("author"),
            "pipeline_tag": metadata.get("pipeline_tag"),
            "license": metadata.get("license"),
            "library_name": metadata.get("library_name"),
            "downloads_last_30_days": metadata.get("downloads_last_30_days"),
            "likes": metadata.get("likes"),
            "score": hit.get("display_score", hit.get("_score")),
            "url": f"https://huggingface.co/{model_id}" if model_id else None,
        })

    return results


def search_models_basic(user_text, limit=10):
    if not user_text:
        return []

    es = Elasticsearch(ES_URL)

    query = {
        "size": limit,
        "_source": [
            "modelID",
            "author",
            "Metadata.pipeline_tag",
            "Metadata.license",
            "Metadata.library_name",
            "Metadata.downloads_last_30_days",
            "Metadata.likes",
            "Features",
        ],
        "query": {
            "multi_match": {
                "query": user_text,
                "fields": [
                    "modelID^2",
                    "author",
                    "Metadata.pipeline_tag^3",
                    "Features^2",
                    "description",
                ],
            }
        },
    }

    response = es.search(index=INDEX_NAME, body=query)
    return clean_results(response, limit=limit)


def _extract_feature_bundle(user_text):
    from EA_Features import FeatureBundle, FunctionalFeatures, QualityFeatures
    from EB_LLM_Client import LLMClient, LoggingLLMClient
    from EC_EssentialFeatureExtractor import EssentialFeaturesExtractor
    from EC_PreferenceFeatureExtractor import PreferenceFeaturesExtractor
    from EC_QualityFeatureExtractor import QualityFeaturesExtractor
    from EC_FunctionalFeatureExtractor import NounPhraseExtractor

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    log_dir = CRITERIA_DIR / "logs" / "django_feature_search"
    log_dir.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(
        api_key=api_key,
        model_name="gemini-3.1-flash-lite",
        max_retries=2,
        retry_delay_seconds=3.0,
    )

    Elogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(log_dir),
        print_output=False,
        save_file="essential.json",
    )
    Plogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(log_dir),
        print_output=False,
        save_file="preference.json",
    )
    Qlogger = LoggingLLMClient(
        llm_client=llm_client,
        save_dir=str(log_dir),
        print_output=False,
        save_file="quality.json",
    )

    Efeatures = EssentialFeaturesExtractor(Elogger).extract(user_text)
    Pfeatures = PreferenceFeaturesExtractor(Plogger).extract(user_text)

    try:
        Qfeatures = QualityFeaturesExtractor(Qlogger).extract(user_text)
    except Exception:
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


def _make_feature_search_builder(limit=10):
    from EE_Query_Builder_Clean_modified_v3_dedupfix import (
        ESQueryBuilderAdaptive,
        query_mapping,
    )

    feature_weight_groups = {
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

    boost_config = {
        "rank": {
            "max": 70.0,
        },
        "match_mode": {
            "grams_factor": 0.90,
        },
    }

    priority_multipliers = {
        "must": 1.8,
        "strong_prefer": 1.4,
        "prefer": 1.0,
        "avoid": 0.0,
    }

    return ESQueryBuilderAdaptive(
        mapping=query_mapping,
        target_hits=50,
        size=limit,
        feature_weight_groups=feature_weight_groups,
        boost_config=boost_config,
        enable_rank_functions=True,
        enable_quality_dimensions=False,
        enable_feature_locations=False,
        priority_multipliers=priority_multipliers,
        synonym_min_conf=0.50,
        minimum_should_match=1,
        synonym_cache_path=str(CRITERIA_DIR / "synonym_cache.json"),
    )


def search_models_feature_based(user_text, limit=10):
    if not user_text:
        return []

    es = Elasticsearch(ES_URL)

    if not es.indices.exists(index=INDEX_NAME):
        raise RuntimeError(f"Elasticsearch index does not exist: {INDEX_NAME}")

    bundle = _extract_feature_bundle(user_text)

    builder = _make_feature_search_builder(limit=limit)
    prebuilt_groups = builder.precompute_feature_group_cache(bundle)

    response, final_query, final_feature_groups = builder.search(
        es_client=es,
        index=INDEX_NAME,
        features=bundle,
        prebuilt_groups=prebuilt_groups,
        include_explain=False,
    )

    return clean_results(response, limit=limit)
