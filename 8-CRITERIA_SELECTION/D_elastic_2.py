"""
index_and_query_models.py

- Loads an Elasticsearch mapping from a JSON file
- Creates an index with that mapping
- Indexes all JSON files from a folder (each JSON = one document)
- Runs a few example queries and prints the results

Requirements:
    pip install elasticsearch

Adjust the CONFIG section below to your paths and index name.
"""

import os
import json
from typing import Iterable, Dict, Any

from elasticsearch import Elasticsearch, helpers


# ========= CONFIG =========

ES_URL = "http://localhost:9200"  # change if needed
INDEX_NAME = "models_t7"             # name of the ES index
MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T7.json"
DATA_FOLDER = "HF-Models-T7"


# ==========================


def load_mapping(path: str) -> Dict[str, Any]:
    """Load the mapping JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping


def recreate_index(es: Elasticsearch, index_name: str, mapping: Dict[str, Any]) -> None:
    """
    Delete the index if it exists, then create it with the given mapping.
    Assumes mapping is a full object like:
        { "mappings": { "properties": { ... } } }
    """
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' exists – deleting it.")
        es.indices.delete(index=index_name)

    # If your mapping JSON starts with { "mappings": { ... } }
    # you can pass it directly as the body:
    es.indices.create(index=index_name, body=mapping)
    print(f"Created index '{index_name}' with provided mapping.")


def iter_documents_from_folder(folder: str, index_name: str) -> Iterable[Dict[str, Any]]:
    """
    Yield bulk actions for all JSON files in the folder.

    Assumes each JSON file is either:
      - a single dict (one document), or
      - a list of dicts (multiple documents).
    """
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        full_path = os.path.join(folder, filename)

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # single dict
        if isinstance(data, dict):
            yield {
                "_index": index_name,
                "_id": filename,  # use filename as id (or change if you have modelID)
                "_source": data,
            }

        # list of dicts
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                yield {
                    "_index": index_name,
                    "_id": f"{filename}#{i}",
                    "_source": item,
                }


def bulk_index(es: Elasticsearch, index_name: str, folder: str) -> None:
    """Bulk index all JSON docs from the folder into the index."""
    actions = iter_documents_from_folder(folder, index_name)

    success, errors = helpers.bulk(
        es,
        actions,
        stats_only=False,
        raise_on_error=False,  # collect errors instead of raising
    )

    print(f"Successfully indexed: {success} documents")
    if errors:
        print(f"Failed to index: {len(errors)} documents. Showing first 5 errors:")
        for e in errors[:5]:
            action = list(e.values())[0]  # the inner "index" or "create" action
            err = action.get("error", {})
            
            print("-" * 60)
            print(f"ID:      {action.get('_id')}")
            print(f"Status:  {action.get('status')}")
            print(f"Type:    {err.get('type')}")
            # print(f"Reason:  {err.get('reason')}")
            
            caused = err.get("caused_by")
            if caused:
                print(f"Caused by: {caused.get('reason')}")


def example_query_top_quality(es: Elasticsearch, index_name: str) -> None:
    """
    Find top 10 models sorted by:
      1. Functional Suitability score (desc)
      2. Functional Suitability num_LH (desc)
    """

    body = {
        "size": 5,
        "query": {
            "exists": {
                "field": "Quality.Functional Suitability.score"
            }
        },
        "sort": [
            {"Quality.Functional Suitability.score": {"order": "desc"}},
            {"Quality.Functional Suitability.num_LH": {"order": "desc"}}
        ]
    }

    res = es.search(index=index_name, body=body)

    print("\nTop 10 models by (score, then LH):")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        fs = src.get("Quality", {}).get("Functional Suitability", {})

        model_id = src.get("modelID")
        score = fs.get("score")
        lh = fs.get("num_LH")

        print(f"- {model_id!r}, FS score={score}, FS LH={lh}")



def example_query_filters(es: Elasticsearch, index_name: str) -> None:
    """
    Example query:
    - All models with:
        * license = "apache-2.0"
        * pipeline_tag = "text-generation"
        * downloads_last_30_days <= 5000
    """

    body = {
        "size": 5,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"Metadata.license": "apache-2.0"}},
                    {"term": {"Metadata.pipeline_tag": "text-generation"}},
                    {"range": {"Metadata.downloads_last_30_days": {"gte": 10000}}},
                ]
            }
        },
        "sort": [
            {"Metadata.downloads_last_30_days": {"order": "asc"}}  # optional
        ]
    }

    res = es.search(index=index_name, body=body)

    print("\nModels with license=apache-2.0, pipeline_tag=text-generation, downloads_last_30_days >= 10000:")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]

        model_id = src.get("modelID")
        license_ = src.get("Metadata", {}).get("license")
        pipeline = src.get("Metadata", {}).get("pipeline_tag")
        downloads30 = src.get("Metadata", {}).get("downloads_last_30_days")

        print(f"- {model_id!r}, license={license_!r}, pipeline={pipeline!r}, downloads_30d={downloads30}")

def example_query_text_search(es: Elasticsearch, index_name: str) -> None:
    """
    Find models where 'sentence-similarity' appears exactly in:
      - tags (list of keywords)
      - Features (keyword)
      - Metadata.pipeline_tag (keyword)
    Using a multi_match query over multiple fields.
    """

    body = {
        "size": 5,
        "query": {
            "multi_match": {
                "query": "sentence-similarity",
                "fields": [
                    "tags",                  # list of keywords
                    "Features",              # single keyword
                    "Metadata.pipeline_tag"  # single keyword under Metadata
                ],
                "type": "best_fields"
            }
        }
    }

    res = es.search(index=index_name, body=body)

    print("\nModels matching 'sentence-similarity' in tags/Features/pipeline_tag:")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        model_id = src.get("modelID")
        tags = src.get("tags")
        features = src.get("Features")
        pipeline_tag = src.get("Metadata", {}).get("pipeline_tag")
        print(f"- modelID={model_id!r}, tags={tags!r}, Features={features!r}, pipeline_tag={pipeline_tag!r}")



def query_underrated_models(es, index_name: str) -> None:
    body = {
        "size": 5,
        "query": {
            "bool": {
                "filter": [
                    {"range": {"Quality.Reliability.score": {"gte": 0.6}}},
                    {"range": {"Metadata.likes": {"lte": 100}}},
                    {"range": {"Metadata.downloads_all_time": {"lte": 10000}}}
                ]
            }
        },
        "sort": [
            {"Quality.Reliability.score": {"order": "desc"}}
        ]
    }

    res = es.search(index=index_name, body=body)
    print("\nUnderrated models (high Reliability score, low likes/downloads):")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        fs = src.get("Quality", {}).get("Reliability", {})
        print(
            f"- {src.get('modelID')!r}, FS={fs.get('score')}, "
            f"likes={src.get('Metadata', {}).get('likes')}, "
            f"dls={src.get('Metadata', {}).get('downloads_all_time')}"
        )

def query_all_rounder_models(es, index_name: str, min_score: float = 0.1) -> None:
    dims = [
        "Functional Suitability",
        "Compatibility",
        "Performance Efficiency",
        "Reliability",
        "Interaction Capability",
        # "Security",
        "Maintainability",
        # "Flexibility",
    ]

    filters = []
    for dim in dims:
        field = f"Quality.{dim}.score"
        filters.append({"range": {field: {"gte": min_score}}})

    body = {
        "size": 5,
        "query": {
            "bool": {
                "filter": filters
            }
        },
        "sort": [
            {"Quality.Functional Suitability.score": {"order": "desc"}}
        ]
    }

    res = es.search(index=index_name, body=body)
    print(f"\nAll-rounder models (all quality dims >= {min_score}):")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        q = src.get("Quality", {})
        scores = {dim: q.get(dim, {}).get("score") for dim in dims}
        print(f"- {src.get('modelID')!r} -> {scores}")

def query_popular_text_generation(es, index_name: str) -> None:
    body = {
        "size": 5,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"Metadata.pipeline_tag": "text-generation"}},
                    {"exists": {"field": "Quality.Functional Suitability.score"}}
                ]
            }
        },
        "sort": [
            {"Quality.Functional Suitability.score": {"order": "desc"}},
            {"Metadata.downloads_last_30_days": {"order": "desc"}}
        ]
    }

    res = es.search(index=index_name, body=body)
    print("\nPopular & high-quality text-generation models:")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        fs = src.get("Quality", {}).get("Functional Suitability", {})
        print(
            f"- {src.get('modelID')!r}, FS={fs.get('score')}, "
            f"downloads_30d={src.get('Metadata', {}).get('downloads_last_30_days')}"
        )

def query_speech_from_facebook(es, index_name: str) -> None:
    """
    Find models mentioning 'speech translation' or 'specaugment'
    in tags / Features / pipeline_tag
    where author = facebook,
    sorted by likes desc then downloads_last_30_days desc.
    """

    body = {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {"term": {"author": "facebook"}}
                ],
                "should": [
                    # MATCH speech translation
                    {"wildcard": {"tags": {"value": "*speech translation*"}}},
                    {"wildcard": {"Features": {"value": "*Speech Translation*"}}},
                    {"wildcard": {"Metadata.pipeline_tag": {"value": "*Speech Translation*"}}},

                    # MATCH specaugment
                    {"wildcard": {"tags": {"value": "*SpecAugment*"}}},
                    {"wildcard": {"Features": {"value": "*SpecAugment*"}}},
                    {"wildcard": {"Metadata.pipeline_tag": {"value": "*SpecAugment*"}}},
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [
            {"Metadata.likes": {"order": "desc"}},
            {"Metadata.downloads_last_30_days": {"order": "desc"}}
        ]
    }

    res = es.search(index=index_name, body=body)

    print("\nFacebook models with 'speech translation' or 'specaugment':")
    for hit in res["hits"]["hits"]:
        src = hit["_source"]

        author = src.get("author")
        tags = src.get("tags")
        feats = src.get("Features")
        pipeline = src.get("Metadata", {}).get("pipeline_tag")
        likes = src.get("Metadata", {}).get("likes")
        downloads30 = src.get("Metadata", {}).get("downloads_last_30_days")
        lastModified = src.get("Metadata", {}).get("lastModified")

        print(
            f"- {src.get('modelID')!r}, author={author!r}, "
            f"likes={likes}, downloads_30d={downloads30}, "
            f"tags={tags}, Features={feats}, pipeline_tag={pipeline}"
            f"lastModified={lastModified}"
        )


def main():
    # 1) connect to Elasticsearch
    es = Elasticsearch(ES_URL)

    # 2) load mapping and recreate index
    mapping = load_mapping(MAPPING_FILE)
    recreate_index(es, INDEX_NAME, mapping)

    # 3) bulk index all documents from folder
    bulk_index(es, INDEX_NAME, DATA_FOLDER)

    # 4) run a few example queries
    print("= = = = =")
    example_query_top_quality(es, INDEX_NAME)
    print("= = = = =")
    example_query_filters(es, INDEX_NAME)
    print("= = = = =")
    example_query_text_search(es, INDEX_NAME)
    print("= = = = =")

    query_underrated_models(es, INDEX_NAME)
    print("= = = = =")
    query_all_rounder_models(es, INDEX_NAME)
    print("= = = = =")
    query_popular_text_generation(es, INDEX_NAME)
    print("= = = = =")

    query_speech_from_facebook(es, INDEX_NAME)
    print("= = = = =")


if __name__ == "__main__":
    main()
