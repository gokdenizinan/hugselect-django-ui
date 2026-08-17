#!/usr/bin/env python3
"""Export and restore a compact, auditable Elasticsearch index snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


FORMAT = "hugselect-elasticsearch-ndjson-v1"


def request_json(url, method="GET", payload=None, content_type="application/json"):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        if isinstance(payload, bytes):
            data = payload
        else:
            data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = content_type

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=120) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Elasticsearch request failed: {exc.code} {detail}") from exc

    return json.loads(body) if body else {}


def clean_settings(raw_settings, index):
    settings = raw_settings[index]["settings"]["index"]
    allowed = {
        key: settings[key]
        for key in (
            "number_of_shards",
            "number_of_replicas",
            "analysis",
            "max_result_window",
        )
        if key in settings
    }
    allowed["number_of_replicas"] = "0"
    return allowed


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_exists(base, encoded_index):
    request = Request(f"{base}/{encoded_index}", method="HEAD")
    try:
        with urlopen(request, timeout=30) as response:
            return response.status == 200
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def export_index(args):
    base = args.url.rstrip("/")
    encoded_index = quote(args.index, safe="")
    mapping = request_json(f"{base}/{encoded_index}/_mapping")
    settings = request_json(f"{base}/{encoded_index}/_settings")
    expected_count = request_json(f"{base}/{encoded_index}/_count")["count"]

    header = {
        "format": FORMAT,
        "index": args.index,
        "document_count": expected_count,
        "settings": clean_settings(settings, args.index),
        "mappings": mapping[args.index]["mappings"],
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    scroll_id = None

    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, separators=(",", ":")) + "\n")
        page = request_json(
            f"{base}/{encoded_index}/_search?scroll=2m",
            method="POST",
            payload={"size": args.batch_size, "sort": ["_doc"], "query": {"match_all": {}}},
        )

        while True:
            scroll_id = page.get("_scroll_id")
            hits = page.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                record = {"_id": hit["_id"], "_source": hit.get("_source", {})}
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported += len(hits)
            print(f"Exported {exported}/{expected_count}", file=sys.stderr)

            page = request_json(
                f"{base}/_search/scroll",
                method="POST",
                payload={"scroll": "2m", "scroll_id": scroll_id},
            )

    if scroll_id:
        request_json(
            f"{base}/_search/scroll",
            method="DELETE",
            payload={"scroll_id": [scroll_id]},
        )

    if exported != expected_count:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Exported {exported} documents; expected {expected_count}.")

    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(f"Created {output} ({exported} documents)")
    print(f"Created {checksum_path}")


def bulk_import(base, index, records):
    body = bytearray()
    for record in records:
        body.extend(
            json.dumps(
                {"index": {"_index": index, "_id": record["_id"]}},
                separators=(",", ":"),
            ).encode("utf-8")
        )
        body.extend(b"\n")
        body.extend(
            json.dumps(record["_source"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        body.extend(b"\n")

    response = request_json(
        f"{base}/_bulk",
        method="POST",
        payload=bytes(body),
        content_type="application/x-ndjson",
    )
    if response.get("errors"):
        failures = [
            item
            for item in response.get("items", [])
            if item.get("index", {}).get("error")
        ]
        raise RuntimeError(f"Bulk import failed: {failures[:3]}")


def import_index(args):
    base = args.url.rstrip("/")
    input_path = Path(args.input).resolve()
    encoded_index = quote(args.index, safe="")

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("format") != FORMAT:
            raise RuntimeError("Unsupported snapshot format.")

        exists = index_exists(base, encoded_index)
        if exists:
            if not args.replace:
                raise RuntimeError(
                    f"Index {args.index!r} already exists. Use --replace to overwrite it."
                )
            request_json(f"{base}/{encoded_index}", method="DELETE")

        request_json(
            f"{base}/{encoded_index}",
            method="PUT",
            payload={"settings": header["settings"], "mappings": header["mappings"]},
        )

        imported = 0
        batch = []
        for line in handle:
            if not line.strip():
                continue
            batch.append(json.loads(line))
            if len(batch) >= args.batch_size:
                bulk_import(base, args.index, batch)
                imported += len(batch)
                batch = []
                print(f"Imported {imported}/{header['document_count']}", file=sys.stderr)

        if batch:
            bulk_import(base, args.index, batch)
            imported += len(batch)

    request_json(f"{base}/{encoded_index}/_refresh", method="POST")
    actual_count = request_json(f"{base}/{encoded_index}/_count")["count"]
    expected_count = header["document_count"]
    if imported != expected_count or actual_count != expected_count:
        raise RuntimeError(
            "Snapshot validation failed: "
            f"read={imported}, indexed={actual_count}, expected={expected_count}."
        )
    print(f"Imported and verified {actual_count} documents in {args.index}.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export an index")
    export_parser.add_argument("--url", default="http://localhost:9200")
    export_parser.add_argument("--index", default="models_t7")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--batch-size", type=int, default=500)
    export_parser.set_defaults(func=export_index)

    import_parser = subparsers.add_parser("import", help="Import an index")
    import_parser.add_argument("--url", default="http://localhost:9200")
    import_parser.add_argument("--index", default="models_t7")
    import_parser.add_argument("--input", required=True)
    import_parser.add_argument("--batch-size", type=int, default=500)
    import_parser.add_argument("--replace", action="store_true")
    import_parser.set_defaults(func=import_index)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
