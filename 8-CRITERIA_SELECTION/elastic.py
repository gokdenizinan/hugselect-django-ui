# models_index_and_query.py
import os
import json
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError


INDEX_NAME = "models_5"
JSON_FOLDER = "HF-Models-T8"
MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T8.json"
# elastic_debug_bulk.py



def create_index_with_mapping(es: Elasticsearch, index_name: str, mapping_file: str) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    with open(mapping_file, "r", encoding="utf-8") as f:
        body = json.load(f)

    es.indices.create(index=index_name, body=body)


def iter_documents_from_folder(folder: str, index_name: str):
    """
    Yield actions for helpers.bulk.
    Attach filename as _id so we can see which docs fail.
    """
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # support single dict or list of dicts
        if isinstance(data, dict):
            yield {
                "_index": index_name,
                "_id": filename,  # helps debugging failures
                "_source": data,
            }
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    yield {
                        "_index": index_name,
                        "_id": f"{filename}#{i}",
                        "_source": item,
                    }


def bulk_index_folder(es: Elasticsearch, index_name: str, folder: str) -> None:
    """
    Bulk index with detailed error logging, so you can see WHY 271 documents failed.
    """
    actions = iter_documents_from_folder(folder, index_name)

    try:
        # raise_on_error=False -> we get success count + list of errors
        success, errors = helpers.bulk(
            es,
            actions,
            stats_only=False,
            raise_on_error=False,
        )
        print(f"Successfully indexed: {success} documents")
        if errors:
            print(f"Failed to index {len(errors)} documents. Showing first 5 errors:")
            for e in errors[:5]:
                print(json.dumps(e, indent=2))
    except BulkIndexError as e:
        # if raise_on_error=True is used somewhere, this catches it
        print(f"BulkIndexError: {len(e.errors)} documents failed.")
        for err in e.errors[:5]:
            print(json.dumps(err, indent=2))


def test_single_document(es: Elasticsearch, index_name: str, json_path: str) -> None:
    """
    Use this to test a single JSON file that you suspect is failing.
    It will print the exact Elasticsearch error.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        resp = es.index(index=index_name, document=data)
        print("Single doc indexed OK:", resp)
    except Exception as e:
        # Elasticsearch will return a detailed error (mapper_parsing_exception, etc.)
        print("Single doc failed to index:")
        print(e)


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    # 1) Create index with mapping (comment out if index already exists and is correct)
    # create_index_with_mapping(es, INDEX_NAME, MAPPING_FILE)

    # 2) Bulk index with detailed error output
    bulk_index_folder(es, INDEX_NAME, JSON_FOLDER)

    # 3) Optional: manually test one file that failed
    # test_single_document(es, INDEX_NAME, "/path/to/suspect/file.json")
