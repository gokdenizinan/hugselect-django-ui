import logging

from django.shortcuts import render
from django.http import  Http404

from .services import (
    build_model_graph,
    get_model_by_id,
    search_models_basic,
    search_models_feature_based,
)

logger = logging.getLogger(__name__)

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
    return render(
        request,
        "recommender/search.html",
        {
            "query": query,
            "results": results,
            "warning": warning,
            "error": error,
            "search_mode": search_mode,
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

    return render(
        request,
        "recommender/decision_stress.html",
        {
            "model_ids": model_ids,
            "search_context": request.session.get(
                "hugselect_last_search",
                {},
            ),
        },
    )