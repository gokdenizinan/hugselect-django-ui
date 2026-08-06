import logging

from django.shortcuts import render, redirect
from django.http import  Http404
from .services import (
    build_model_graph,
    get_model_by_id,
    search_models_basic,
    search_models_feature_based,
)
from .decision_stress import (
    collect_essential_requirements,
    run_decision_stress_test,
    summarize_decision_stress_test,
)

logger = logging.getLogger(__name__)

def _normalize_base_models(value):
    """
    Return a clean, unique list of base-model names.
    """

    if value is None:
        return []

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        candidates = [value]

    normalized = []

    for candidate in candidates:
        base_model = str(candidate).strip()

        if base_model and base_model not in normalized:
            normalized.append(base_model)

    return normalized


def group_search_results_by_base_model(results):
    """
    Group ranked search results without changing their order.
    """

    single_groups_by_name = {}
    single_base_model_groups = []
    multiple_base_model_results = []
    no_base_model_results = []

    for position, result in enumerate(results, start=1):

        result["tooltip_key"] = f"result-{position}"

        base_models = _normalize_base_models(
            result.get("basemodels")
        )

        if not base_models:
            no_base_model_results.append(result)
            continue

        if len(base_models) > 1:
            multiple_base_model_results.append(result)
            continue

        base_model = base_models[0]

        if base_model not in single_groups_by_name:
            group = {
                "base_model": base_model,
                "results": [],
            }

            single_groups_by_name[base_model] = group
            single_base_model_groups.append(group)

        single_groups_by_name[
            base_model
        ]["results"].append(result)

    return {
        "single_base_model_groups": single_base_model_groups,
        "multiple_base_model_results": (
            multiple_base_model_results
        ),
        "no_base_model_results": no_base_model_results,
    }
def search_view(request):
    query = ""
    results = []
    warning = None
    error = None
    search_mode = None

    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            try:
                results = search_models_feature_based(query, limit=10)
                search_mode = "feature-based"

            except Exception:
                logger.exception(
                    "Feature-based search failed; trying basic fallback."
                )

                try:
                    results = search_models_basic(query, limit=10)
                    search_mode = "basic-fallback"
                    warning = (
                        "Advanced recommendation is temporarily unavailable. "
                        "Showing basic search results instead."
                    )

                except Exception:
                    logger.exception(
                        "Basic fallback search also failed."
                    )
                    error = (
                        "Model search is temporarily unavailable. "
                        "Please check that Elasticsearch is running and try again."
                    )
        else:
            error = "Please describe the model you need."

    if results and search_mode:
        request.session["hugselect_last_search"] = {
            "query": query,
            "search_mode": search_mode,
            "scores": {
                result["model_id"]: result.get("score")
                for result in results
                if result.get("model_id")
            },
            "explanations": {
                result["model_id"]: result.get("match_explanation")
                for result in results
                if result.get("model_id") and result.get("match_explanation")
            },
        }
    if request.method == "POST":
        request.session["hugselect_search_results"] = {
            "query": query,
            "results": results,
            "warning": warning,
            "error": error,
            "search_mode": search_mode,
        }

        return redirect("search_results")
    grouped_results = group_search_results_by_base_model(
        results
    )
    return render(
        request,
        "recommender/search.html",
        {
            "query": query,
            "results": results,
            "grouped_results": grouped_results,
            "warning": warning,
            "error": error,
            "search_mode": search_mode,
        },
    )
def search_results_view(request):
    """
    Display the most recently completed model search.
    """

    search_state = request.session.get(
        "hugselect_search_results",
        {},
    )

    results = search_state.get("results", [])


    grouped_results = group_search_results_by_base_model(
        results
    )

    return render(
        request,
        "recommender/search_results.html",
        {
            "query": search_state.get("query", ""),
            "results": results,
            "grouped_results": grouped_results,
            "warning": search_state.get("warning"),
            "error": search_state.get("error"),
            "search_mode": search_state.get(
                "search_mode"
            ),
        },
    )

def model_detail_view(request, model_id):
    model = get_model_by_id(model_id)
    if model is None:
        raise Http404("Model not found.")
    graph = build_model_graph(model)
    last_search = request.session.get("hugselect_last_search", {})
    explanations = last_search.get("explanations", {})
    scores = last_search.get("scores", {})

    search_context = {
        "query": last_search.get("query"),
        "search_mode": last_search.get("search_mode"),
        "score": scores.get(model_id),
        "explanation": explanations.get(model_id),
    }

    return render(
        request,
        "recommender/model_detail.html",
        {
            "model": model,
            "search_context": search_context,
            "graph": graph,
        },
    )

def compare_models_view(request):
    model_ids = list(
        dict.fromkeys(request.GET.getlist("model_ids"))
    )

    if len(model_ids) < 2 or len(model_ids) > 3:
        return render(
            request,
            "recommender/compare.html",
            {
                "error": "Please select two or three models to compare.",
                "models": [],
            },
        )

    models = []

    for model_id in model_ids:
        model = get_model_by_id(model_id)

        if model is not None:
            models.append(model)

    if len(models) < 2:
        return render(
            request,
            "recommender/compare.html",
            {
                "error": (
                    "At least two of the selected models "
                    "could not be found."
                ),
                "models": models,
            },
        )

    last_search = request.session.get(
        "hugselect_last_search",
        {},
    )

    search_mode = last_search.get("search_mode")
    scores = last_search.get("scores", {})
    explanations = last_search.get("explanations", {})

    for model in models:
        model_id = model["model_id"]

        for field_name in ("language", "basemodels"):
                value = model.get(field_name)

                if isinstance(value, list):
                    model[field_name] = ", ".join(
                        str(item) for item in value
                    )

        model["comparison_score"] = scores.get(model_id)
        model["comparison_explanation"] = explanations.get(model_id)
        model["is_strongest_match"] = False

    if search_mode == "feature-based":
        scored_models = [
            model
            for model in models

            if isinstance(
                model.get("comparison_score"),
                (int, float),
            )
        ]

        if scored_models:
            strongest_score = max(
                model["comparison_score"]
                for model in scored_models
            )

            strongest_models = [
                model
                for model in scored_models
                if model["comparison_score"] == strongest_score
            ]

            strongest_label = (
                "Joint strongest match"
                if len(strongest_models) > 1
                else "Strongest overall match"
            )

            for model in strongest_models:
                model["is_strongest_match"] = True
                model["strongest_label"] = strongest_label
    coverage_rows = []

    if search_mode == "feature-based":
        rows_by_requirement = {}

        for model in models:
            explanation = (
                model.get("comparison_explanation") or {}
            )

            for feature_group in explanation.get(
                "per_feature",
                [],
            ):
                for match in feature_group.get("matches", []):
                    feature_key = match.get("feature_key")
                    user_value = match.get("user_value")

                    if not feature_key:
                        continue

                    requirement_key = (
                        feature_key,
                        str(user_value),
                    )

                    if requirement_key not in rows_by_requirement:
                        rows_by_requirement[requirement_key] = {
                            "feature_key": feature_key,
                            "user_value": user_value,
                            "statuses": {},
                        }

                    rows_by_requirement[
                        requirement_key
                    ]["statuses"][model["model_id"]] = match.get(
                        "matched",
                        False,
                    )

        for row in rows_by_requirement.values():
            row["model_statuses"] = [
                {
                    "model_id": model["model_id"],
                    "matched": row["statuses"].get(
                        model["model_id"]
                    ),
                }
                for model in models
            ]

            coverage_rows.append(row)

    def numeric_value(value):
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    decision_summary = None

    if search_mode == "feature-based" and coverage_rows:
        scored_models = [
            model
            for model in models
            if isinstance(
                model.get("comparison_score"),
                (int, float),
            )
        ]

        if scored_models:
            best_score = max(
                model["comparison_score"]
                for model in scored_models
            )

            score_leaders = [
                model
                for model in scored_models
                if model["comparison_score"] == best_score
            ]

            tie_breaker_model = None

            if len(score_leaders) > 1:
                best_popularity = max(
                    (
                        numeric_value(
                            model.get("downloads_last_30_days")
                        ),
                        numeric_value(model.get("likes")),
                    )
                    for model in score_leaders
                )

                popularity_leaders = [
                    model
                    for model in score_leaders
                    if (
                        numeric_value(
                            model.get("downloads_last_30_days")
                        ),
                        numeric_value(model.get("likes")),
                    ) == best_popularity
                ]

                if len(popularity_leaders) == 1:
                    tie_breaker_model = popularity_leaders[0]["model_id"]

            decision_summary = {
                "leaders": [
                    model["model_id"]
                    for model in score_leaders
                ],
                "best_score": best_score,
                "is_tie": len(score_leaders) > 1,
                "tie_breaker_model": tie_breaker_model,
            }
    return render(
        request,
        "recommender/compare.html",
        {
            "models": models,
            "coverage_rows": coverage_rows,
            "decision_summary": decision_summary,
            "search_context": {
                "query": last_search.get("query"),
                "search_mode": search_mode,
            },
        },
    )
def decision_stress_view(request):
    model_ids = list(
        dict.fromkeys(request.GET.getlist("model_ids"))
    )

    if len(model_ids) < 2 or len(model_ids) > 3:
        return render(
            request,
            "recommender/decision_stress.html",
            {
                "error": (
                    "Please select two or three models "
                    "to run a decision stress test."
                ),
                "models": [],
            },
        )

    last_search = request.session.get(
        "hugselect_last_search",
        {},
    )

    search_mode = last_search.get("search_mode")
    explanations = last_search.get("explanations", {})

    if search_mode != "feature-based":
        return render(
            request,
            "recommender/decision_stress.html",
            {
                "error": (
                    "Decision stress testing is available only "
                    "for feature-based recommendations."
                ),
                "models": [],
            },
        )

    model_explanations = {
        model_id: explanations[model_id]
        for model_id in model_ids
        if explanations.get(model_id)
    }
    essential_requirements = collect_essential_requirements(
    model_explanations
)

    missing_explanation_ids = [
        model_id
        for model_id in model_ids
        if model_id not in model_explanations
    ]

    if len(model_explanations) < 2:
        return render(
            request,
            "recommender/decision_stress.html",
            {
                "error": (
                    "Structured explanations are unavailable for "
                    "at least two selected models."
                ),
                "models": [],
            },
        )

    stress_result = run_decision_stress_test(
        model_explanations
    )

    stress_summary = summarize_decision_stress_test(
        stress_result
    )

    return render(
        request,
        "recommender/decision_stress.html",
        {
            "model_ids": model_ids,
            "search_context": last_search,
            "stress_result": stress_result,
            "stress_summary": stress_summary,
            "essential_requirements": essential_requirements,
            "missing_explanation_ids": missing_explanation_ids,
        },
    )
