import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from elasticsearch import Elasticsearch


ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"
DISPLAY_RESULT_LIMIT = 10
AVAILABILITY_CANDIDATE_LIMIT = 30
AVAILABILITY_HTTP_TIMEOUT_SECONDS = 3.0
AVAILABILITY_MAX_WORKERS = 5
BASE_MODEL_FAMILY_PATTERNS = {
    "llama": "*llama*",
    "mistral": "*mistral*",
    "qwen": "*qwen*",
    "gemma": "*gemma*",
}

EXPLICIT_REQUIREMENT_FEATURE_GROUPS = (
    (
        "Core",
        (
            ("task", "Task"),
            ("domain", "Domain"),
            ("author", "Author"),
            ("objective", "Objective"),
            ("model_name", "Model name"),
        ),
    ),
    (
        "Metadata",
        (
            ("license_name", "License"),
            ("library_name", "Library"),
            ("basemodels", "Base models"),
            ("datasets", "Datasets"),
            ("language", "Language"),
            ("metrics", "Metrics"),
            ("gated", "Gated"),
            ("last_modified_year", "Last modified year"),
        ),
    ),
    (
        "Functional",
        (("functional", "Functional requirement"),),
    ),
    (
        "Quality",
        (
            ("Functional_Suitability", "Functional suitability"),
            ("Compatibility", "Compatibility"),
            ("Performance_Efficiency", "Performance efficiency"),
            ("Reliability", "Reliability"),
            ("Interaction_Capability", "Interaction capability"),
            ("Security", "Security"),
            ("Maintainability", "Maintainability"),
            ("Flexibility", "Flexibility"),
        ),
    ),
)
EXPLICIT_REQUIREMENT_FEATURE_LABELS = {
    feature_key: label
    for _, options in EXPLICIT_REQUIREMENT_FEATURE_GROUPS
    for feature_key, label in options
}
EXPLICIT_REQUIREMENT_PRIORITIES = (
    ("must", "MUST"),
    ("should", "SHOULD"),
    ("could", "COULD"),
    ("wont", "WON'T"),
)
EXPLICIT_REQUIREMENT_PRIORITY_LABELS = dict(
    EXPLICIT_REQUIREMENT_PRIORITIES
)


def parse_explicit_requirements(
    feature_keys,
    values,
    priorities,
):
    """Validate and normalize explicit MoSCoW form rows."""
    row_count = max(len(feature_keys), len(values), len(priorities), 0)
    requirements = []
    seen = {}

    for index in range(row_count):
        feature_key = (
            feature_keys[index].strip()
            if index < len(feature_keys)
            else ""
        )
        value = values[index].strip() if index < len(values) else ""
        priority = (
            priorities[index].strip().casefold()
            if index < len(priorities)
            else ""
        )

        if not feature_key and not value:
            continue

        row_number = index + 1
        if feature_key not in EXPLICIT_REQUIREMENT_FEATURE_LABELS:
            raise ValueError(
                f"Requirement {row_number} has an unsupported feature."
            )
        if not value:
            raise ValueError(
                f"Requirement {row_number} must include a value."
            )
        if priority not in EXPLICIT_REQUIREMENT_PRIORITY_LABELS:
            raise ValueError(
                f"Requirement {row_number} has an unsupported priority."
            )

        canonical_key = (
            feature_key,
            " ".join(
                re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    value.casefold(),
                ).split()
            ),
        )
        if canonical_key in seen:
            previous_priority = seen[canonical_key]
            feature_label = EXPLICIT_REQUIREMENT_FEATURE_LABELS[
                feature_key
            ]
            if previous_priority == priority:
                raise ValueError(
                    f"Duplicate requirement: {feature_label} = {value}."
                )
            raise ValueError(
                "Contradictory requirements: "
                f"{feature_label} = {value} cannot be both "
                f"{EXPLICIT_REQUIREMENT_PRIORITY_LABELS[previous_priority]} "
                f"and {EXPLICIT_REQUIREMENT_PRIORITY_LABELS[priority]}."
            )

        seen[canonical_key] = priority
        requirements.append({
            "feature_key": feature_key,
            "value": value,
            "priority": priority,
        })

    return requirements


def check_model_availability(
    url,
    timeout=AVAILABILITY_HTTP_TIMEOUT_SECONDS,
):
    """Return the current availability status for a model page."""

    if not url:
        return {
            "status": "unknown",
            "http_status": None,
        }

    request = Request(
        url,
        headers={
            "User-Agent": "HugSelect availability check",
            "Accept": "text/html",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            http_status = response.status
    except HTTPError as exc:
        http_status = exc.code
    except (TimeoutError, URLError, OSError):
        return {
            "status": "unknown",
            "http_status": None,
        }

    if http_status == 200:
        status = "available"
    elif http_status in (404, 410):
        status = "unavailable"
    else:
        status = "unknown"

    return {
        "status": status,
        "http_status": http_status,
    }


def _result_availability_url(result):
    url = result.get("url")

    if url:
        return url

    model_id = result.get("model_id")

    if model_id:
        return f"https://huggingface.co/{model_id}"

    return None


def select_available_results(
    ranked_results,
    display_limit=DISPLAY_RESULT_LIMIT,
    availability_checker=None,
    max_workers=AVAILABILITY_MAX_WORKERS,
):
    """
    Keep the first verified-available results in their ranked order.

    Checks run in small batches so request time is bounded without
    continuing through the whole candidate pool after enough results
    have been selected. Duplicate URLs are checked only once.
    """

    if availability_checker is None:
        availability_checker = check_model_availability

    candidates = list(ranked_results)
    selected_results = []
    availability_by_url = {}
    summary = {
        "candidate_count": len(candidates),
        "checked_count": 0,
        "http_check_count": 0,
        "available_count": 0,
        "unavailable_count": 0,
        "unknown_count": 0,
        "display_limit": display_limit,
        "shortfall": display_limit,
    }

    if display_limit <= 0 or not candidates:
        return selected_results, summary

    worker_count = max(1, int(max_workers))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for batch_start in range(0, len(candidates), worker_count):
            if len(selected_results) >= display_limit:
                break

            batch = candidates[
                batch_start:batch_start + worker_count
            ]
            urls = [
                _result_availability_url(result)
                for result in batch
            ]
            pending_checks = {}

            for url in urls:
                if (
                    url
                    and url not in availability_by_url
                    and url not in pending_checks
                ):
                    pending_checks[url] = executor.submit(
                        availability_checker,
                        url,
                    )

            summary["http_check_count"] += len(pending_checks)

            for url, future in pending_checks.items():
                try:
                    availability = future.result()
                except Exception:
                    availability = {
                        "status": "unknown",
                        "http_status": None,
                    }

                if not isinstance(availability, dict):
                    availability = {
                        "status": "unknown",
                        "http_status": None,
                    }

                if availability.get("status") not in {
                    "available",
                    "unavailable",
                    "unknown",
                }:
                    availability = {
                        "status": "unknown",
                        "http_status": availability.get(
                            "http_status"
                        ),
                    }

                availability_by_url[url] = availability

            for result, url in zip(batch, urls):
                if len(selected_results) >= display_limit:
                    break

                availability = availability_by_url.get(
                    url,
                    {
                        "status": "unknown",
                        "http_status": None,
                    },
                )
                status = availability["status"]

                summary["checked_count"] += 1
                summary[f"{status}_count"] += 1

                if status == "available":
                    retained_result = dict(result)
                    retained_result["availability"] = dict(
                        availability
                    )
                    selected_results.append(retained_result)

    summary["shortfall"] = max(
        0,
        display_limit - len(selected_results),
    )

    return selected_results, summary


def normalize_base_model_family(base_model_family):
    """Return a supported normalized family key, or None."""

    normalized_family = str(
        base_model_family or ""
    ).strip().casefold()

    if normalized_family in BASE_MODEL_FAMILY_PATTERNS:
        return normalized_family

    return None


def _base_model_family_filter(
    base_model_family,
):
    """
    Build an Elasticsearch candidate filter for a model family.
    """

    normalized_family = normalize_base_model_family(
        base_model_family,
    )

    if normalized_family is None:
        return None

    pattern = BASE_MODEL_FAMILY_PATTERNS[
        normalized_family
    ]

    return {
        "wildcard": {
            "Metadata.basemodels": {
                "value": pattern,
                "case_insensitive": True,
            }
        }
    }

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
            "language": metadata.get("language"),
            "datasets": metadata.get("datasets"),
            "basemodels": metadata.get("basemodels"),
            "model_type": metadata.get("model_type"),
            "metrics": metadata.get("metrics"),
            "tags": metadata.get("tags"),
            "quality": src.get("Quality") or {},
            "downloads_last_30_days": metadata.get("downloads_last_30_days"),
            "likes": metadata.get("likes"),
            "score": hit.get("display_score", hit.get("_score")),
            "match_explanation": hit.get("match_explanation"),
            "moscow_requirements": hit.get("moscow_requirements"),
            "explicit_requirements": hit.get("explicit_requirements"),
            "url": f"https://huggingface.co/{model_id}" if model_id else None,
        })

    return results


def search_models_basic(
    user_text,
    limit=10,
    base_model_family=None,
):
    if not user_text:
        return []

    es = Elasticsearch(ES_URL)

    text_query = {
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
    }
    family_filter = _base_model_family_filter(
        base_model_family
    )

    query_clause = text_query

    if family_filter is not None:
        query_clause = {
            "bool": {
                "must": [text_query],
                "filter": [family_filter],
            }
        }

    query = {
        "size": limit,
        "_source": [
            "modelID",
            "author",
            "Metadata.pipeline_tag",
            "Metadata.license",
            "Metadata.library_name",
            "Metadata.basemodels",
            "Metadata.downloads_last_30_days",
            "Metadata.likes",
            "Features",
        ],
        "query": query_clause,
    }

    response = es.search(index=INDEX_NAME, body=query)
    return clean_results(response, limit=limit)

def get_model_by_id(model_id):
    if not model_id:
        return None

    es = Elasticsearch(ES_URL)

    query = {
        "size": 1,
        "query": {
            "term": {
                "modelID": model_id,
            }
        },
    }

    response = es.search(index=INDEX_NAME, body=query)
    results = clean_results(response, limit=1)

    return results[0] if results else None

def build_model_graph(model):
    if not model:
        return {"nodes": [], "edges": []}

    nodes = [
        {
            "id": "model",
            "label": model["model_id"],
            "type": "model",
        }
    ]
    edges = []

    relationships = [
        ("author", model.get("author"), "Created by"),
        ("task", model.get("pipeline_tag"), "Performs"),
        ("license", model.get("license"), "Licensed under"),
        ("library", model.get("library_name"), "Uses"),
        ("base_model", model.get("basemodels"), "Based on"),
        ("model_type", model.get("model_type"), "Model type"),
    ]

    languages = model.get("language") or []
    if isinstance(languages, str):
        languages = [languages]

    for index, language in enumerate(languages):
        relationships.append(
            (f"language_{index}", language, "Supports")
        )

    relationships = [
        (
            node_id,
            ", ".join(map(str, value)) if isinstance(value, list) else value,
            relationship,
        )
        for node_id, value, relationship in relationships
    ]
    for node_id, value, relationship in relationships:
        if not value:
            continue

        nodes.append({
            "id": node_id,
            "label": str(value),
            "type": {
            "base_model": "base_model",
            "model_type": "model_type",
        }.get(node_id, node_id.split("_")[0]),
        })

        edges.append({
            "source": "model",
            "target": node_id,
            "label": relationship,
        })

    return {
        "nodes": nodes,
        "edges": edges,
    }

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
        temperature=0.0,
        seed=0,
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


def search_models_feature_based(
    user_text,
    limit=10,
    base_model_family=None,
    explicit_requirements=None,
):
    if not user_text:
        return []

    total_start = time.perf_counter()

    es = Elasticsearch(ES_URL)

    index_check_start = time.perf_counter()
    if not es.indices.exists(index=INDEX_NAME):
        raise RuntimeError(f"Elasticsearch index does not exist: {INDEX_NAME}")
    index_check_seconds = time.perf_counter() - index_check_start

    feature_extraction_start = time.perf_counter()
    bundle = _extract_feature_bundle(user_text)
    feature_extraction_seconds = (
        time.perf_counter() - feature_extraction_start
    )

    builder_creation_start = time.perf_counter()
    builder = _make_feature_search_builder(limit=limit)
    builder_creation_seconds = (
        time.perf_counter() - builder_creation_start
    )

    precompute_start = time.perf_counter()
    prebuilt_groups = builder.precompute_feature_group_cache(bundle)
    if explicit_requirements:
        prebuilt_groups = builder.apply_explicit_requirements(
            prebuilt_groups,
            explicit_requirements,
        )
    precompute_seconds = time.perf_counter() - precompute_start

    search_start = time.perf_counter()
    family_filter = _base_model_family_filter(
        base_model_family
    )

    extra_filter_clauses = (
        [family_filter]
        if family_filter is not None
        else None
    )
    response, final_query, final_feature_groups = builder.search(
        es_client=es,
        index=INDEX_NAME,
        features=bundle,
        prebuilt_groups=prebuilt_groups,
        include_explain=False,
        extra_filter_clauses=extra_filter_clauses,
    )
    search_seconds = time.perf_counter() - search_start

    explanations_start = time.perf_counter()
    moscow_requirements = builder.summarize_moscow_requirements(
        final_feature_groups
    )
    feasible_hits = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {}) or {}

        hit["match_explanation"] = builder.compare_bundle_to_sample(
            bundle,
            source,
            prebuilt_groups=final_feature_groups,
        )
        if not hit["match_explanation"]["hard_filters_passed"]:
            continue
        hit["moscow_requirements"] = moscow_requirements
        hit["explicit_requirements"] = explicit_requirements or []
        feasible_hits.append(hit)
    response.setdefault("hits", {})["hits"] = feasible_hits
    explanations_seconds = time.perf_counter() - explanations_start

    total_seconds = time.perf_counter() - total_start

    print(
        "Feature search timings | "
        f"index_check={index_check_seconds:.3f}s | "
        f"feature_extraction={feature_extraction_seconds:.3f}s | "
        f"builder_creation={builder_creation_seconds:.3f}s | "
        f"precompute={precompute_seconds:.3f}s | "
        f"search={search_seconds:.3f}s | "
        f"explanations={explanations_seconds:.3f}s | "
        f"total={total_seconds:.3f}s"
    )

    return clean_results(response, limit=limit)
