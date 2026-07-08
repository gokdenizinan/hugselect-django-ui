from elasticsearch import Elasticsearch


ES_URL = "http://localhost:9200"
INDEX_NAME = "models_t7"


def clean_results(response, limit=10):
    hits = response.get("hits", {}).get("hits", [])
    results = []

    for hit in hits[:limit]:
        src = hit.get("_source", {}) or {}
        metadata = src.get("Metadata", {}) or {}

        model_id = src.get("modelID")

        results.append({
            "model_id": model_id,
            "author": src.get("author"),
            "pipeline_tag": metadata.get("pipeline_tag"),
            "downloads_last_30_days": metadata.get("downloads_last_30_days"),
            "likes": metadata.get("likes"),
            "score": hit.get("_score"),
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