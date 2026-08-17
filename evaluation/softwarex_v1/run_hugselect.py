#!/usr/bin/env python3
"""Run the reproducible, API-independent HugSelect fallback for all scenarios."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hugselect_web"))

from recommender.services import INDEX_NAME, ES_URL, search_models_basic  # noqa: E402


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        default=str(Path(__file__).with_name("scenarios.json")),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    scenario_data = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    runs = []
    for scenario in scenario_data["scenarios"]:
        started = time.perf_counter()
        results = search_models_basic(scenario["query"], limit=args.limit)
        runs.append(
            {
                "scenario_id": scenario["id"],
                "system": "HugSelect",
                "system_label": "HugSelect v1.0.0 basic fallback",
                "run_at": datetime.now(timezone.utc).isoformat(),
                "browsing_enabled": False,
                "fresh_conversation": True,
                "follow_up_prompts": 0,
                "status": "completed",
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "query": scenario["query"],
                "model_ids": [row["model_id"] for row in results],
                "raw_results": results,
            }
        )
        print(f"{scenario['id']}: {len(results)} candidates")

    payload = {
        "protocol_version": scenario_data["protocol_version"],
        "release_commit": git_sha(),
        "elasticsearch_url": ES_URL,
        "elasticsearch_index": INDEX_NAME,
        "search_mode": "basic-fallback",
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
