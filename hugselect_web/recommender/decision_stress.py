from __future__ import annotations


ESSENTIAL_FEATURE_KEYS = {
    "task",
    "task_alias",
    "domain",
    "domain_alias",
    "author",
    "objective",
    "model_name",
}

PREFERENCE_FEATURE_KEYS = {
    "license_name",
    "library_name",
    "basemodels",
    "datasets",
    "language",
    "metrics",
    "gated",
    "last_modified_year",
}

FUNCTIONAL_FEATURE_KEYS = {
    "functional",
    "functional_item",
}

QUALITY_FEATURE_KEYS = {
    "Functional_Suitability",
    "Compatibility",
    "Performance_Efficiency",
    "Reliability",
    "Interaction_Capability",
    "Security",
    "Maintainability",
    "Flexibility",
}


SCENARIOS = (
    {
        "key": "current",
        "label": "Current weights",
        "description": "Uses the normal HugSelect requirement weights.",
        "multipliers": {
            "essential": 1.0,
            "preference": 1.0,
            "functional": 1.0,
            "quality": 1.0,
        },
        "strict_essentials": False,
    },
    {
        "key": "essential_first",
        "label": "Essential requirements first",
        "description": (
            "Doubles the influence of essential requirements."
        ),
        "multipliers": {
            "essential": 2.0,
            "preference": 0.75,
            "functional": 1.0,
            "quality": 1.0,
        },
        "strict_essentials": False,
    },
    {
        "key": "preference_first",
        "label": "Preferences first",
        "description": (
            "Doubles the influence of license, language, "
            "library, and related preferences."
        ),
        "multipliers": {
            "essential": 1.0,
            "preference": 2.0,
            "functional": 0.75,
            "quality": 1.0,
        },
        "strict_essentials": False,
    },
    {
        "key": "functional_first",
        "label": "Functional requirements first",
        "description": (
            "Doubles the influence of requested capabilities."
        ),
        "multipliers": {
            "essential": 1.0,
            "preference": 0.75,
            "functional": 2.0,
            "quality": 1.0,
        },
        "strict_essentials": False,
    },
    {
        "key": "strict_essentials",
        "label": "Strict essentials",
        "description": (
            "Prioritizes essentials and penalizes models "
            "that miss essential requirements."
        ),
        "multipliers": {
            "essential": 2.0,
            "preference": 0.75,
            "functional": 0.75,
            "quality": 1.0,
        },
        "strict_essentials": True,
    },
)


def category_for_feature(feature_key: str) -> str:
    """Return the requirement category for an explanation feature."""

    if feature_key in ESSENTIAL_FEATURE_KEYS:
        return "essential"

    if feature_key in PREFERENCE_FEATURE_KEYS:
        return "preference"

    if feature_key in FUNCTIONAL_FEATURE_KEYS:
        return "functional"

    if feature_key in QUALITY_FEATURE_KEYS:
        return "quality"

    return "unknown"

def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _requirement_key(feature_key: str, user_value) -> str:
    value_text = str(user_value).strip().casefold()

    if value_text:
        return value_text

    return f"feature:{feature_key}"


def score_explanation_for_scenario(
    explanation: dict,
    scenario: dict,
) -> dict:
    """
    Rescore one model's existing match explanation under one scenario.

    This does not call Gemini or Elasticsearch.
    """

    multipliers = scenario.get("multipliers", {})

    best_by_requirement = {}
    essential_statuses = {}
    unknown_features = set()

    for feature_group in explanation.get("per_feature", []):
        group_weight = feature_group.get(
            "effective_weight",
            0.0,
        )

        for match in feature_group.get("matches", []):
            feature_key = str(
                match.get("feature_key") or ""
            )

            category = category_for_feature(feature_key)

            if category == "unknown":
                if feature_key:
                    unknown_features.add(feature_key)
                continue

            multiplier = _safe_float(
                multipliers.get(category, 1.0)
            )

            effective_weight = _safe_float(
                match.get("effective_weight", group_weight)
            )

            original_score = _safe_float(
                match.get("score")
            )

            requirement_key = _requirement_key(
                feature_key,
                match.get("user_value"),
            )

            candidate = {
                "feature_key": feature_key,
                "category": category,
                "user_value": match.get("user_value"),
                "matched": bool(match.get("matched")),
                "actual": original_score * multiplier,
                "possible": effective_weight * multiplier,
            }

            current = best_by_requirement.get(
                requirement_key
            )

            if (
                current is None
                or candidate["actual"] > current["actual"]
            ):
                best_by_requirement[
                    requirement_key
                ] = candidate

            if category == "essential":
                existing_status = essential_statuses.get(
                    requirement_key
                )

                if existing_status is None:
                    essential_statuses[requirement_key] = {
                        "feature_key": feature_key,
                        "user_value": match.get("user_value"),
                        "matched": bool(match.get("matched")),
                    }
                else:
                    existing_status["matched"] = (
                        existing_status["matched"]
                        or bool(match.get("matched"))
                    )

    total_actual = sum(
        item["actual"]
        for item in best_by_requirement.values()
    )

    total_possible = sum(
        item["possible"]
        for item in best_by_requirement.values()
    )

    normalized_score = (
        min(100.0, (total_actual / total_possible) * 100.0)
        if total_possible > 0
        else 0.0
    )

    missed_essentials = [
        {
            "feature_key": status["feature_key"],
            "user_value": status["user_value"],
        }
        for status in essential_statuses.values()
        if not status["matched"]
    ]

    strict_exclusion = (
        bool(scenario.get("strict_essentials"))
        and bool(missed_essentials)
    )

    if strict_exclusion:
        normalized_score = 0.0

    category_contributions = {
        "essential": 0.0,
        "preference": 0.0,
        "functional": 0.0,
        "quality": 0.0,
    }

    for item in best_by_requirement.values():
        category_contributions[item["category"]] += item[
            "actual"
        ]

    return {
        "scenario_key": scenario.get("key"),
        "score": round(normalized_score, 2),
        "raw_score": round(total_actual, 4),
        "maximum_score": round(total_possible, 4),
        "strict_exclusion": strict_exclusion,
        "missed_essentials": missed_essentials,
        "category_contributions": category_contributions,
        "unknown_features": sorted(unknown_features),
    }
def run_decision_stress_test(
    model_explanations: dict[str, dict],
) -> dict:
    """
    Score all selected models under every stress-test scenario.
    """

    scenario_results = []

    for scenario in SCENARIOS:
        model_results = []

        for model_id, explanation in model_explanations.items():
            result = score_explanation_for_scenario(
                explanation,
                scenario,
            )

            model_results.append({
                "model_id": model_id,
                **result,
            })

        model_results.sort(
            key=lambda item: (
                -item["score"],
                item["model_id"].casefold(),
            )
        )

        best_score = (
            model_results[0]["score"]
            if model_results
            else 0.0
        )

        winners = [
            result["model_id"]
            for result in model_results
            if result["score"] == best_score
        ]

        current_rank = 0
        previous_score = None

        for position, result in enumerate(
            model_results,
            start=1,
        ):
            if result["score"] != previous_score:
                current_rank = position
                previous_score = result["score"]

            result["rank"] = current_rank

        scenario_results.append({
            "key": scenario["key"],
            "label": scenario["label"],
            "description": scenario["description"],
            "winners": winners,
            "is_tie": len(winners) > 1,
            "model_results": model_results,
        })

    return {
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
    }

def summarize_decision_stress_test(
    stress_result: dict,
) -> dict:
    """
    Summarize outright wins and ties across all tested scenarios.
    """

    scenarios = stress_result.get("scenarios", [])
    scenario_count = len(scenarios)

    outright_win_counts = {}
    tied_scenario_count = 0

    for scenario in scenarios:
        winners = scenario.get("winners", [])

        if len(winners) != 1:
            tied_scenario_count += 1
            continue

        winner = winners[0]
        outright_win_counts[winner] = (
            outright_win_counts.get(winner, 0) + 1
        )

    if scenario_count == 0:
        return {
            "stability_percentage": 0.0,
            "stability_label": "No result",
            "leaders": [],
            "win_counts": {},
            "highest_win_count": 0,
            "scenario_count": 0,
            "tied_scenario_count": 0,
            "is_tied": False,
            "all_scenarios_tied": False,
        }

    all_scenarios_tied = (
        tied_scenario_count == scenario_count
    )

    if all_scenarios_tied:
        joint_winners = sorted({
            model_id
            for scenario in scenarios
            for model_id in scenario.get("winners", [])
        })

        return {
            "stability_percentage": 0.0,
            "stability_label": "Evidence-equivalent models",
            "leaders": joint_winners,
            "win_counts": {
                model_id: 0
                for model_id in joint_winners
            },
            "highest_win_count": 0,
            "scenario_count": scenario_count,
            "tied_scenario_count": tied_scenario_count,
            "is_tied": True,
            "all_scenarios_tied": True,
        }

    highest_win_count = max(
        outright_win_counts.values(),
        default=0,
    )

    leaders = sorted(
        model_id
        for model_id, wins in outright_win_counts.items()
        if wins == highest_win_count
    )

    stability_percentage = round(
        (highest_win_count / scenario_count) * 100.0,
        1,
    )

    if len(leaders) > 1:
        stability_label = "No consistent winner"
    elif stability_percentage >= 80.0:
        stability_label = "Stable winner"
    elif stability_percentage >= 60.0:
        stability_label = "Moderately sensitive"
    else:
        stability_label = "Highly sensitive"

    return {
        "stability_percentage": stability_percentage,
        "stability_label": stability_label,
        "leaders": leaders,
        "win_counts": outright_win_counts,
        "highest_win_count": highest_win_count,
        "scenario_count": scenario_count,
        "tied_scenario_count": tied_scenario_count,
        "is_tied": len(leaders) > 1,
        "all_scenarios_tied": False,
    }
def collect_essential_requirements(
    model_explanations: dict[str, dict],
) -> list[dict]:
    """
    Collect the unique requirements classified as essential
    in the stored match explanations.
    """

    requirements_by_key = {}

    for explanation in model_explanations.values():
        for feature_group in explanation.get(
            "per_feature",
            [],
        ):
            for match in feature_group.get("matches", []):
                feature_key = str(
                    match.get("feature_key") or ""
                )

                if category_for_feature(feature_key) != "essential":
                    continue

                user_value = match.get("user_value")

                requirement_key = _requirement_key(
                    feature_key,
                    user_value,
                )

                requirements_by_key.setdefault(
                    requirement_key,
                    {
                        "key": requirement_key,
                        "feature_key": feature_key,
                        "user_value": user_value,
                    },
                )

    return sorted(
        requirements_by_key.values(),
        key=lambda requirement: (
            requirement["feature_key"].casefold(),
            str(requirement["user_value"]).casefold(),
        ),
    )