import json
from pathlib import Path
from pprint import pprint

from elasticsearch import Elasticsearch

from EA_Features import (
    EssentialFeatures,
    PreferenceFeatures,
    QualityFeatures,
    FeatureBundle,
    FunctionalFeatures,
)

from EC_FunctionalFeatureExtractor import NounPhraseExtractor

from EC_PreferenceFeatureExtractor import (
    to_categorical_feat,
    to_numeric_feat,
    to_bool_feat,
    to_recency_feat,
)

from E_utils import parse_llm_json_flex, object_to_dict

from test_live_search import (
    ES_URL,
    INDEX_NAME,
    make_builder,
    print_hits,
)


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "logs" / "live_search"


USER_TEXT = (
    "I need a text classification model for Turkish sentiment analysis, "
    "preferably based on BERT and usable commercially."
)


def load_cached_response_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        if not raw:
            raise RuntimeError(f"Empty cache file: {path}")
        record = raw[-1]
    elif isinstance(raw, dict):
        record = raw
    else:
        raise RuntimeError(f"Unexpected cache format in {path}: {type(raw).__name__}")

    response_text = record.get("response")
    if not response_text:
        raise RuntimeError(f"No response field found in cache file: {path}")

    return parse_llm_json_flex(response_text)


def add_last_modified_aliases(preferences):
    recency_value = None

    for attr_name in ("last_modified_year", "last_modified", "lastModified"):
        if hasattr(preferences, attr_name):
            candidate = getattr(preferences, attr_name)
            if candidate is not None:
                recency_value = candidate
                break

    if recency_value is None:
        return preferences

    for attr_name in ("last_modified_year", "last_modified", "lastModified"):
        try:
            setattr(preferences, attr_name, recency_value)
        except Exception:
            pass

    return preferences


def build_bundle_from_cache() -> FeatureBundle:
    Edata = load_cached_response_json(CACHE_DIR / "essential.json")
    Pdata = load_cached_response_json(CACHE_DIR / "preference.json")
    Qdata = load_cached_response_json(CACHE_DIR / "quality.json")

    Efeatures = EssentialFeatures(
        task=to_categorical_feat(Edata.get("task")),
        domain=to_categorical_feat(Edata.get("domain")),
        model_name=to_categorical_feat(Edata.get("model_name")),
        author=to_categorical_feat(Edata.get("author")),
        objective=to_categorical_feat(Edata.get("objective")),
        task_aliases=to_categorical_feat(Edata.get("task_aliases")),
        domain_aliases=to_categorical_feat(Edata.get("domain_aliases")),
    )

    Pfeatures = PreferenceFeatures(
        basemodels=to_categorical_feat(Pdata.get("basemodels")),
        license_name=to_categorical_feat(Pdata.get("license_name")),
        downloads_all_time=to_numeric_feat(Pdata.get("downloads_all_time")),
        downloads_last_30_days=to_numeric_feat(Pdata.get("downloads_last_30_days")),
        file_count=to_numeric_feat(Pdata.get("file_count")),
        gated=to_bool_feat(Pdata.get("gated")),
        lastModified=to_recency_feat(Pdata.get("lastModified")),
        library_name=to_categorical_feat(Pdata.get("library_name")),
        likes=to_numeric_feat(Pdata.get("likes")),
        tensors_total=to_numeric_feat(Pdata.get("tensors_total")),
        usedStorage=to_numeric_feat(Pdata.get("usedStorage")),
        datasets=to_categorical_feat(Pdata.get("datasets")),
        language=to_categorical_feat(Pdata.get("language")),
        metrics=to_categorical_feat(Pdata.get("metrics")),
    )
    Pfeatures = add_last_modified_aliases(Pfeatures)

    Qfeatures = QualityFeatures(
        Functional_Suitability=Qdata.get("Functional_Suitability"),
        Compatibility=Qdata.get("Compatibility"),
        Performance_Efficiency=Qdata.get("Performance_Efficiency"),
        Reliability=Qdata.get("Reliability"),
        Interaction_Capability=Qdata.get("Interaction_Capability"),
        Security=Qdata.get("Security"),
        Maintainability=Qdata.get("Maintainability"),
        Flexibility=Qdata.get("Flexibility"),
    )

    Ffeatures = FunctionalFeatures()
    Fextractor = NounPhraseExtractor()
    Ffeatures.add_from_query(USER_TEXT, Fextractor)

    return FeatureBundle(
        essential=Efeatures,
        preferences=Pfeatures,
        quality=Qfeatures,
        functional=Ffeatures,
    )


def show_object(title, obj):
    print(f"\n--- {title} ---")
    try:
        pprint(object_to_dict(obj), width=120)
    except Exception:
        pprint(obj, width=120)


def main():
    print("\nUSER TEXT:")
    print(USER_TEXT)

    print("\n[1] Building FeatureBundle from cached LLM logs...")
    bundle = build_bundle_from_cache()
    show_object("FeatureBundle", bundle)

    print("\n[2] Checking alias candidate files...")
    alias_dir = BASE_DIR / "alias_candidates"
    required = [
        "pipeline_tag.json",
        "tags.json",
        "model_type.json",
        "Features.json",
        "license.json",
        "library_name.json",
        "basemodels.json",
        "datasets.json",
        "language.json",
        "metrics.json",
        "gated.json",
        "author.json",
    ]
    missing = [name for name in required if not (alias_dir / name).exists()]
    if missing:
        raise RuntimeError(
            "Missing alias candidate files:\n"
            + "\n".join(missing)
            + "\nRun the minimal alias_candidates creation step first."
        )
    print("Alias candidate files exist.")

    print("\n[3] Connecting to Elasticsearch...")
    es_client = Elasticsearch(ES_URL)

    if not es_client.indices.exists(index=INDEX_NAME):
        raise RuntimeError(f"Elasticsearch index does not exist: {INDEX_NAME}")

    print(f"Connected. Index exists: {INDEX_NAME}")

    print("\n[4] Building query and searching...")
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

    print("\nDONE: cached FeatureBundle + Elasticsearch search worked.")


if __name__ == "__main__":
    main()
