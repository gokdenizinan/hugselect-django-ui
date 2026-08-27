#!/usr/bin/env python3
"""Run reproducible field ablations of HugSelect's basic fallback retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hugselect_web"))

from elasticsearch import Elasticsearch  # noqa: E402
from recommender.services import ES_URL, INDEX_NAME  # noqa: E402


VARIANTS = {
    "baseline": [
        "modelID^2",
        "author",
        "Metadata.pipeline_tag^3",
        "Features^2",
        "description",
    ],
    "without_functional_features": [
        "modelID^2",
        "author",
        "Metadata.pipeline_tag^3",
        "description",
    ],
    "without_metadata_boost": [
        "modelID^2",
        "author",
        "Metadata.pipeline_tag",
        "Features^2",
        "description",
    ],
    "without_model_id": [
        "author",
        "Metadata.pipeline_tag^3",
        "Features^2",
        "description",
    ],
    "uniform_fields": [
        "modelID",
        "author",
        "Metadata.pipeline_tag",
        "Features",
        "description",
    ],
    "without_description": [
        "modelID^2",
        "author",
        "Metadata.pipeline_tag^3",
        "Features^2",
    ],
    "task_and_features_only": [
        "Metadata.pipeline_tag^3",
        "Features^2",
    ],
    "description_only": [
        "description",
    ],
}


def retrieve(es, query, fields, limit):
    body = {
        "size": limit,
        "_source": ["modelID"],
        "query": {"multi_match": {"query": query, "fields": fields}},
    }
    response = es.search(index=INDEX_NAME, body=body)
    return [
        hit.get("_source", {}).get("modelID") or hit.get("_id")
        for hit in response.get("hits", {}).get("hits", [])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=str(Path(__file__).with_name("scenarios.json")),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))["scenarios"]
    es = Elasticsearch(ES_URL)
    runs = []
    for scenario in scenarios:
        variant_results = {
            name: retrieve(es, scenario["query"], fields, args.limit)
            for name, fields in VARIANTS.items()
        }
        baseline = variant_results["baseline"]
        baseline_set = set(baseline)
        metrics = {}
        for name, candidates in variant_results.items():
            overlap = len(baseline_set.intersection(candidates))
            metrics[name] = {
                "top_1_changed": baseline[:1] != candidates[:1],
                "overlap_at_10": overlap,
                "jaccard_at_10": round(
                    overlap / len(baseline_set.union(candidates)), 4
                    if baseline_set.union(candidates)
                    else 1.0,
                ),
            }
        runs.append(
            {
                "scenario_id": scenario["id"],
                "query": scenario["query"],
                "variants": variant_results,
                "metrics_against_baseline": metrics,
            }
        )

    summary = {}
    for name in VARIANTS:
        values = [run["metrics_against_baseline"][name] for run in runs]
        summary[name] = {
            "top_1_change_rate": round(
                sum(value["top_1_changed"] for value in values) / len(values), 4
            ),
            "mean_overlap_at_10": round(
                sum(value["overlap_at_10"] for value in values) / len(values), 3
            ),
            "mean_jaccard_at_10": round(
                sum(value["jaccard_at_10"] for value in values) / len(values), 4
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "scope": "HugSelect basic-fallback multi_match field ablation",
                "variants": VARIANTS,
                "summary": summary,
                "runs": runs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
