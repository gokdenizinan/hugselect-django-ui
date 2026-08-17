#!/usr/bin/env python3
"""Validate model IDs and summarize the SoftwareX comparison runs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


USER_AGENT = "HugSelect-SoftwareX-evaluation/1.0"


def load_runs(directory):
    runs = []
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.extend(payload.get("runs", []))
    return runs


def fetch_model(model_id):
    url = f"https://huggingface.co/api/models/{quote(model_id, safe='/')}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            card_data = payload.get("cardData") or {}
            base_model = card_data.get("base_model")
            if isinstance(base_model, str):
                base_models = [base_model]
            elif isinstance(base_model, list):
                base_models = [str(value) for value in base_model]
            else:
                base_models = []
            return {
                "valid": True,
                "canonical_id": payload.get("id") or model_id,
                "pipeline_tag": payload.get("pipeline_tag"),
                "library_name": payload.get("library_name"),
                "base_models": base_models,
                "gated": payload.get("gated"),
            }
        except HTTPError as exc:
            if exc.code == 404:
                return {
                    "valid": False,
                    "http_status": 404,
                    "resolution": "not_found",
                }
            if exc.code == 401:
                return {
                    "valid": False,
                    "http_status": 401,
                    "resolution": "not_publicly_resolvable",
                }
            if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except (URLError, TimeoutError) as exc:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    page_request = Request(
        f"https://huggingface.co/{quote(model_id, safe='/')}",
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
    )
    try:
        with urlopen(page_request, timeout=20) as response:
            if response.status == 200:
                return {
                    "valid": True,
                    "canonical_id": model_id,
                    "pipeline_tag": None,
                    "library_name": None,
                    "base_models": [],
                    "resolution": "public_model_page",
                }
    except HTTPError as exc:
        if exc.code in {401, 404}:
            return {
                "valid": False,
                "http_status": exc.code,
                "resolution": (
                    "not_found" if exc.code == 404 else "not_publicly_resolvable"
                ),
            }
        return {"valid": None, "http_status": exc.code}
    except (URLError, TimeoutError) as exc:
        return {"valid": None, "error": type(exc).__name__}
    return {"valid": None, "error": "unresolved"}


def bootstrap_ci(values, seed=20260817, draws=10000):
    if not values:
        return None
    if len(values) == 1:
        return [round(values[0], 4), round(values[0], 4)]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(draws)
    )
    return [
        round(means[int(0.025 * draws)], 4),
        round(means[int(0.975 * draws) - 1], 4),
    ]


def cohen_kappa(judgments):
    grouped = defaultdict(dict)
    for item in judgments:
        key = (item["scenario_id"], item["system"], item["model_id"])
        grouped[key][item["reviewer_id"]] = int(item["relevant"])
    reviewer_ids = sorted({item["reviewer_id"] for item in judgments})
    if len(reviewer_ids) < 2:
        return None
    first, second = reviewer_ids[:2]
    pairs = [
        (scores[first], scores[second])
        for scores in grouped.values()
        if first in scores and second in scores
    ]
    if not pairs:
        return None
    observed = sum(a == b for a, b in pairs) / len(pairs)
    p1 = sum(a for a, _ in pairs) / len(pairs)
    p2 = sum(b for _, b in pairs) / len(pairs)
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"reviewers": [first, second], "items": len(pairs), "kappa": round(kappa, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reviews",
        default=str(Path(__file__).with_name("reviews.json")),
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    runs = load_runs(args.runs)
    completed = [run for run in runs if run.get("status") == "completed"]
    model_ids = sorted(
        {model_id for run in completed for model_id in run.get("model_ids", [])},
        key=str.casefold,
    )

    cache_path = Path(args.output).with_name("hf_model_cache.json")
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    for state in cache.values():
        if state.get("valid") is None and state.get("http_status") == 401:
            state.update(
                valid=False,
                resolution="not_publicly_resolvable",
            )
    missing = [
        model_id
        for model_id in model_ids
        if model_id not in cache
        or (
            cache[model_id].get("valid") is None
            and cache[model_id].get("http_status") == 429
        )
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_model, model_id): model_id for model_id in missing}
        for index, future in enumerate(as_completed(futures), start=1):
            model_id = futures[future]
            cache[model_id] = future.result()
            if index % 25 == 0 or index == len(missing):
                print(f"Validated {index}/{len(missing)} new identifiers")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    by_system = defaultdict(list)
    for run in runs:
        by_system[run["system"]].append(run)

    hugselect_by_scenario = {
        run["scenario_id"]: run
        for run in by_system.get("HugSelect", [])
        if run.get("status") == "completed"
    }

    summaries = {}
    for system, system_runs in sorted(by_system.items()):
        completed_runs = [run for run in system_runs if run.get("status") == "completed"]
        scenario_valid_rates = []
        valid_count = invalid_count = unknown_count = candidate_count = 0
        not_found_count = not_public_count = 0
        overlaps = []
        jaccards = []
        family_overlaps = []

        for run in completed_runs:
            ids = run.get("model_ids", [])
            candidate_count += len(ids)
            states = [cache.get(model_id, {"valid": None}) for model_id in ids]
            checked = [state for state in states if state.get("valid") is not None]
            valid = sum(state.get("valid") is True for state in states)
            invalid = sum(state.get("valid") is False for state in states)
            unknown = len(states) - valid - invalid
            valid_count += valid
            invalid_count += invalid
            unknown_count += unknown
            not_found_count += sum(
                state.get("resolution") == "not_found"
                for state in states
            )
            not_public_count += sum(
                state.get("resolution") == "not_publicly_resolvable"
                for state in states
            )
            if checked:
                scenario_valid_rates.append(valid / len(checked))

            baseline = hugselect_by_scenario.get(run["scenario_id"])
            if baseline and system != "HugSelect":
                ours = {value.casefold() for value in baseline.get("model_ids", [])}
                theirs = {value.casefold() for value in ids}
                intersection = len(ours.intersection(theirs))
                union = len(ours.union(theirs))
                overlaps.append(intersection)
                jaccards.append(intersection / union if union else 0.0)

                our_families = {
                    family.casefold()
                    for value in baseline.get("model_ids", [])
                    for family in cache.get(value, {}).get("base_models", [])
                }
                their_families = {
                    family.casefold()
                    for value in ids
                    for family in cache.get(value, {}).get("base_models", [])
                }
                if our_families or their_families:
                    family_overlaps.append(len(our_families.intersection(their_families)))

        checked_total = valid_count + invalid_count
        summaries[system] = {
            "system_label": next((run.get("system_label") for run in system_runs), None),
            "scenario_records": len(system_runs),
            "completed_runs": len(completed_runs),
            "unavailable_or_failed_runs": len(system_runs) - len(completed_runs),
            "candidate_ids": candidate_count,
            "valid_ids": valid_count,
            "invalid_or_inaccessible_ids": invalid_count,
            "nonexistent_ids": not_found_count,
            "not_publicly_resolvable_ids": not_public_count,
            "unknown_validation_ids": unknown_count,
            "valid_id_rate": round(valid_count / checked_total, 4) if checked_total else None,
            "nonexistent_id_rate": round(not_found_count / checked_total, 4) if checked_total else None,
            "not_publicly_resolvable_rate": round(not_public_count / checked_total, 4) if checked_total else None,
            "valid_id_rate_95pct_bootstrap_ci": bootstrap_ci(scenario_valid_rates),
            "mean_exact_overlap_at_10_with_hugselect": round(statistics.fmean(overlaps), 3) if overlaps else None,
            "mean_exact_jaccard_with_hugselect": round(statistics.fmean(jaccards), 4) if jaccards else None,
            "mean_recorded_base_model_overlap_with_hugselect": round(statistics.fmean(family_overlaps), 3) if family_overlaps else None,
        }

    review_payload = json.loads(Path(args.reviews).read_text(encoding="utf-8"))
    output_payload = {
        "analysis_notes": [
            "Identifier existence was checked against the public Hugging Face model API.",
            "Exact-ID and recorded-base-model overlap measure agreement, not correctness.",
            "Unavailable systems are retained as unavailable rather than imputed.",
            "Confidence intervals resample scenario-level validity rates with a fixed seed.",
        ],
        "systems": summaries,
        "inter_reviewer_agreement": cohen_kappa(review_payload.get("judgments", [])),
        "run_status_counts": {
            status: sum(run.get("status") == status for run in runs)
            for status in sorted({run.get("status") for run in runs})
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
