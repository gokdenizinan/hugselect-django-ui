# elastic_find_and_strip_conflicts.py
#
# 1) Creates index with mapping
# 2) Tries to bulk index
# 3) Collects all field paths that cause mapping conflicts
# 4) Removes those paths from the mapping dict
# 5) Writes a new cleaned mapping file
#
# After running this, you can:
#   - delete the index
#   - recreate it with the cleaned mapping
#   - bulk index again

import os
import re
import json
from copy import deepcopy
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError


INDEX_NAME = "models"
JSON_FOLDER = "HF-Models-T6"
MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_T6.json"
CLEANED_MAPPING_FILE = "8-CRITERIA_SELECTION/es_mapping_counts_cleaned.json"


def create_index_with_mapping(es: Elasticsearch, index_name: str, mapping_file: str) -> dict:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    with open(mapping_file, "r", encoding="utf-8") as f:
        body = json.load(f)

    resp = es.indices.create(index=index_name, body=body)
    return body  # return mapping dict as loaded


def iter_documents_from_folder(folder: str, index_name: str):
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            yield {
                "_index": index_name,
                "_id": filename,
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


def extract_conflict_path_from_reason(reason: str) -> str | None:
    """
    reason examples:
      "[1:1414] object mapping for [metadata.config.tokenizer_config.bos_token] tried to parse field [bos_token] as object, but found a concrete value"

    We want "metadata.config.tokenizer_config.bos_token".
    """
    matches = re.findall(r"\[([^\]]+)\]", reason)
    # matches might be like: ["1:1414", "metadata.config.tokenizer_config.bos_token", "bos_token"]
    for m in matches:
        # skip location like "1:1414"
        if ":" in m:
            continue
        # pick first that looks like a dotted path
        if "." in m:
            return m
    return None


def collect_conflict_paths_from_errors(errors: list) -> set[str]:
    conflict_paths: set[str] = set()
    for e in errors:
        try:
            err = e.get("index", {}).get("error", {})
            reason = err.get("reason", "")
            path = extract_conflict_path_from_reason(reason)
            if path:
                conflict_paths.add(path)
        except Exception:
            continue
    return conflict_paths


def bulk_index_and_collect_conflicts(es: Elasticsearch, index_name: str, folder: str) -> set[str]:
    actions = iter_documents_from_folder(folder, index_name)
    try:
        success, errors = helpers.bulk(
            es,
            actions,
            stats_only=False,
            raise_on_error=False,
        )
        print(f"Successfully indexed: {success} documents")
        print(f"Failed to index: {len(errors)} documents")

        if errors:
            print("Sample errors (first 3):")
            for e in errors[:3]:
                print(json.dumps(e, indent=2))

        conflict_paths = collect_conflict_paths_from_errors(errors)
        print("\nConflict field paths:")
        for p in sorted(conflict_paths):
            print("  ", p)

        return conflict_paths

    except BulkIndexError as e:
        print(f"BulkIndexError: {len(e.errors)} documents failed.")
        conflict_paths = collect_conflict_paths_from_errors(e.errors)
        for err in e.errors[:3]:
            print(json.dumps(err, indent=2))
        return conflict_paths


def remove_field_from_properties(props: dict, path_parts: list[str]) -> bool:
    """
    Remove a field from a nested mapping.properties dict.
    Returns True if removed, False if not found.
    """
    if not path_parts:
        return False

    key = path_parts[0]

    if len(path_parts) == 1:
        # leaf to delete
        if key in props:
            del props[key]
            return True
        return False

    # not leaf, go deeper in properties
    node = props.get(key)
    if not isinstance(node, dict):
        return False

    # nested object can have 'properties'
    nested_props = node.get("properties")
    if not isinstance(nested_props, dict):
        return False

    removed = remove_field_from_properties(nested_props, path_parts[1:])

    # if nested object now has empty properties, you might optionally clean it up
    if removed and not nested_props:
        # either keep empty object in mapping or delete it:
        # del props[key]
        pass

    return removed


def strip_conflict_paths_from_mapping(mapping_dict: dict, conflict_paths: set[str]) -> dict:
    """
    Given an ES mapping dict like:
      { "mappings": { "properties": { ... } } }
    remove all fields whose dotted paths are in conflict_paths.
    """
    new_mapping = deepcopy(mapping_dict)
    properties = new_mapping.get("mappings", {}).get("properties", {})

    for path in conflict_paths:
        parts = path.split(".")
        removed = remove_field_from_properties(properties, parts)
        print(f"Removing {path}: {'OK' if removed else 'NOT FOUND'}")

    return new_mapping


if __name__ == "__main__":
    es = Elasticsearch("http://localhost:9200")

    # 1) create index with current mapping
    original_mapping = create_index_with_mapping(es, INDEX_NAME, MAPPING_FILE)

    # 2) bulk index and collect conflict field paths
    conflict_paths = bulk_index_and_collect_conflicts(es, INDEX_NAME, JSON_FOLDER)

    # 3) build cleaned mapping with those paths removed
    cleaned_mapping = strip_conflict_paths_from_mapping(original_mapping, conflict_paths)

    # 4) save cleaned mapping JSON
    with open(CLEANED_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nWrote cleaned mapping to: {CLEANED_MAPPING_FILE}")

    # 5) optional: delete index and recreate from cleaned mapping
    # es.indices.delete(index=INDEX_NAME)
    # es.indices.create(index=INDEX_NAME, body=cleaned_mapping)
    # bulk_index_and_collect_conflicts(es, INDEX_NAME, JSON_FOLDER)
