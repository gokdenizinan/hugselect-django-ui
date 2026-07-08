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
        author = src.get("author")
        pipeline_tag = metadata.get("pipeline_tag")
        downloads = metadata.get("downloads_last_30_days")
        likes = metadata.get("likes")
        score = hit.get("_score")

        results.append({
            "model_id": model_id,
            "author": author,
            "pipeline_tag": pipeline_tag,
            "downloads_last_30_days": downloads,
            "likes": likes,
            "score": score,
            "url": f"https://huggingface.co/{model_id}" if model_id else None,
        })

    return results


es = Elasticsearch(ES_URL)

query = {
    "size": 3,
    "_source": [
        "modelID",
        "author",
        "Metadata.pipeline_tag",
        "Metadata.downloads_last_30_days",
        "Metadata.likes",
    ],
    "query": {
        "match": {
            "Metadata.pipeline_tag": "text-generation"
        }
    }
}

response = es.search(index=INDEX_NAME, body=query)
cleaned = clean_results(response)

for i, item in enumerate(cleaned, start=1):
    print(f"\nResult {i}")
    print(f"Model: {item['model_id']}")
    print(f"Author: {item['author']}")
    print(f"Task: {item['pipeline_tag']}")
    print(f"Downloads 30d: {item['downloads_last_30_days']}")
    print(f"Likes: {item['likes']}")
    print(f"Score: {item['score']}")
    print(f"URL: {item['url']}")
