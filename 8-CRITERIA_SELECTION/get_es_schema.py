import os
import json
from collections import defaultdict


# ---------- 1. Collect types per field path ---------- #

def collect_types(value, path, type_stats):
    """
    Recursively walk the JSON structure and record the Python types
    of non-null scalar values at each dotted path (e.g. "user.name").
    """
    # Dive into nested dictionaries
    if isinstance(value, dict):
        for k, v in value.items():
            new_path = f"{path}.{k}" if path else k
            collect_types(v, new_path, type_stats)

    # Arrays: just iterate and reuse the same path
    elif isinstance(value, list):
        for item in value:
            collect_types(item, path, type_stats)

    else:
        # Scalar (or None)
        if value is not None and path:
            type_stats[path].add(type(value))


def scan_folder_for_types(folder_path):
    """
    Load all JSON files in folder_path and collect type information
    for each field path.
    """
    type_stats = defaultdict(set)
    file_count = 0

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename}: error loading JSON ({e})")
            continue

        file_count += 1

        # Support top-level dict or list of dicts
        if isinstance(data, dict):
            collect_types(data, "", type_stats)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    collect_types(item, "", type_stats)
        else:
            print(f"Skipping {filename}: top-level JSON is {type(data).__name__}, not dict or list")

    if file_count == 0:
        print("No valid JSON files found.")
    else:
        print(f"Scanned {file_count} JSON files.\n")

    return type_stats


# ---------- 2. Infer Elasticsearch field types ---------- #

def infer_es_type(py_types):
    """
    Map a set of Python types to an Elasticsearch field type.
    Very simple heuristic — adjust as needed.
    """
    # Remove NoneType if present (we don't add it, but just in case)
    py_types = {t for t in py_types if t is not type(None)}

    if not py_types:
        return None  # no non-null values seen

    # One single type
    if len(py_types) == 1:
        t = next(iter(py_types))
        if t is bool:
            return "boolean"
        if t is int:
            return "long"
        if t is float:
            return "double"
        if t is str:
            # You might prefer "text" with a "keyword" subfield; this is simpler:
            return "keyword"
        # Fallback
        return "keyword"

    # Mixed numeric types -> treat as double
    numeric_types = {int, float}
    if py_types.issubset(numeric_types):
        return "double"

    # Mixed or unknown -> safest generic
    return "keyword"


# ---------- 3. Build nested ES mapping structure ---------- #

def insert_field(mapping_properties, path, es_type):
    """
    Given a dotted path like "user.address.city" and an ES type like "keyword",
    insert it into the nested "properties" dict structure.
    """
    parts = path.split(".")
    current = mapping_properties

    for i, part in enumerate(parts):
        is_leaf = (i == len(parts) - 1)

        if is_leaf:
            # Leaf field definition
            field_def = current.setdefault(part, {})
            # Only set type if not set yet
            field_def.setdefault("type", es_type)
        else:
            # Intermediate object
            node = current.setdefault(part, {})
            # Turn it into an object with properties if not already
            if "properties" not in node:
                node["type"] = "object"
                node["properties"] = {}
            current = node["properties"]


def build_es_mapping(type_stats):
    """
    Build an Elasticsearch mapping object from collected type_stats.
    Only includes fields that have at least one non-null value.
    """
    root_properties = {}

    for path, py_types in type_stats.items():
        es_type = infer_es_type(py_types)
        if es_type is None:
            # No non-null values ever seen for this field -> skip it
            continue
        insert_field(root_properties, path, es_type)

    mapping = {
        "mappings": {
            "properties": root_properties
        }
    }
    return mapping

def collect_mapping_keys(mapping, prefix="", out=None):
    if out is None:
        out = []

    props = mapping.get("properties", {})
    for key, value in props.items():
        path = f"{prefix}.{key}" if prefix else key
        out.append(path)

        # recurse into nested objects
        if isinstance(value, dict) and "properties" in value:
            collect_mapping_keys(value, path, out)

    return out



# ---------- 4. Putting it all together ---------- #

if __name__ == "__main__":
    # 👉 Change this to your folder path
    folder = "HF-Models-T4"

    type_stats = scan_folder_for_types(folder)
    es_mapping = build_es_mapping(type_stats)
    output_file = "8-CRITERIA_SELECTION/es_mapping.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(es_mapping, f, indent=2, ensure_ascii=False)

    print(f"Saved mapping to {output_file}")

    # Pretty-print the mapping JSON
    print("Elasticsearch mapping (excluding fields that are always null):\n")
    print(json.dumps(es_mapping, indent=2, ensure_ascii=False))

    all_keys = collect_mapping_keys(es_mapping["mappings"])

    key_file = "8-CRITERIA_SELECTION/mapping_keys.json"

    with open(key_file, "w", encoding="utf-8") as f:
        json.dump(all_keys, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_keys)} keys to {output_file}")


