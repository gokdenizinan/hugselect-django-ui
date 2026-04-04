import os
import json
from typing import Dict, Any, Iterable, Tuple, List
from elasticsearch import Elasticsearch, helpers

# OLMADI BU YAPAMADIM
class ESClient:
    def __init__(
        self,
        es_url: str,
        index_name: str,
        mapping_file: str,
        data_folder: str,
    ) -> None:
        self.es_url = es_url
        self.index_name = index_name
        self.mapping_file = mapping_file
        self.data_folder = data_folder

        # Lazily you could inject this, but simplest is to create it here
        self.es = Elasticsearch(self.es_url)

    # -------------------------------
    # Mapping / index management
    # -------------------------------
    def load_mapping(self) -> Dict[str, Any]:
        """Load the mapping JSON file configured for this indexer."""
        with open(self.mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping

    def recreate_index(self, mapping: Dict[str, Any]) -> None:
        """
        Delete the index if it exists, then create it with the given mapping.

        Assumes mapping is a full object like:
            { "mappings": { "properties": { ... } } }
        """
        if self.es.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' exists – deleting it.")
            self.es.indices.delete(index=self.index_name)

        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"Created index '{self.index_name}' with provided mapping.")

    # -------------------------------
    # Document iteration
    # -------------------------------
    def iter_documents_from_folder(self) -> Iterable[Dict[str, Any]]:
        """
        Yield bulk actions for all JSON files in the configured data folder.

        Assumes each JSON file is either:
          - a single dict (one document), or
          - a list of dicts (multiple documents).
        """
        for filename in os.listdir(self.data_folder):
            if not filename.endswith(".json"):
                continue

            full_path = os.path.join(self.data_folder, filename)

            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # single dict
            if isinstance(data, dict):
                yield {
                    "_index": self.index_name,
                    "_id": filename,  # use filename as id (or change if you have modelID)
                    "_source": data,
                }

            # list of dicts
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                    yield {
                        "_index": self.index_name,
                        "_id": f"{filename}#{i}",
                        "_source": item,
                    }

    # -------------------------------
    # Bulk indexing
    # -------------------------------
    def bulk_index(self) -> Tuple[int, list]:
        """Bulk index all JSON docs from the folder into the index."""
        actions = self.iter_documents_from_folder()

        success, errors = helpers.bulk(
            self.es,
            actions,
            stats_only=False,
            raise_on_error=False,  # collect errors instead of raising
        )

        print(f"Successfully indexed: {success} documents")
        if errors:
            print(f"Failed to index: {len(errors)} documents. Showing first 5 errors:")
            for e in errors[:5]:
                print(json.dumps(e, indent=2))

        return success, errors
    
    def search(self, size: int = 5) -> List[Dict[str, Any]]:
        """
        Query models authored by 'facebook' that mention:
        - speech translation
        - specaugment

        in tags / Features / Metadata.pipeline_tag.

        Results sorted by:
          - likes desc
          - downloads_last_30_days desc
        """

        result = self.es.search(index=self.index_name, body=body)

        print("\nFacebook models mentioning 'speech translation' or 'specaugment':")
        hits = result.get("hits", {}).get("hits", [])

        for hit in hits:
            src = hit["_source"]
            meta = src.get("Metadata", {})

            print(
                f"- {src.get('modelID')!r}, "
                f"author={src.get('author')!r}, "
                f"likes={meta.get('likes')}, "
                f"downloads_30d={meta.get('downloads_last_30_days')}, "
                f"tags={src.get('tags')}, "
                f"Features={src.get('Features')}, "
                f"pipeline_tag={meta.get('pipeline_tag')}"
            )

        return hits
    

body = {
    "size": 2,
    "query": {
        "bool": {
            "must": [
                {"term": {"author": "facebook"}}
            ],
            "should": [
                {"wildcard": {"tags": {"value": "*speech translation*"}}},
                {"wildcard": {"Features": {"value": "*Speech Translation*"}}},
                {"wildcard": {"Metadata.pipeline_tag": {"value": "*Speech Translation*"}}},

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

if __name__ == "__main__":
    ES_URL = "http://localhost:9200"  # change if needed
    INDEX_NAME = "models_02"          # name of the ES index
    MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T8.json"
    DATA_FOLDER = "HF-Models-T8"

    indexer = ESIndexer(
        es_url=ES_URL,
        index_name=INDEX_NAME,
        mapping_file=MAPPING_FILE,
        data_folder=DATA_FOLDER,
    )

    mapping = indexer.load_mapping()
    indexer.recreate_index(mapping)
    indexer.bulk_index()
