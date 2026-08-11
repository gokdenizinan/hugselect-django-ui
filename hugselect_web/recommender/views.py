import logging

from django.shortcuts import render, redirect
from django.http import Http404, HttpResponse
from django.utils import timezone
from .services import (
    AVAILABILITY_CANDIDATE_LIMIT,
    DISPLAY_RESULT_LIMIT,
    EXPLICIT_REQUIREMENT_FEATURE_GROUPS,
    EXPLICIT_REQUIREMENT_FEATURE_LABELS,
    EXPLICIT_REQUIREMENT_PRIORITIES,
    build_model_graph,
    get_model_by_id,
    normalize_base_model_family,
    parse_explicit_requirements,
    search_models_basic,
    search_models_feature_based,
    select_available_results,
)
from .decision_stress import (
    collect_essential_requirements,
    run_decision_stress_test,
    summarize_decision_stress_test,
)
from .reporting import generate_analysis_report

logger = logging.getLogger(__name__)

MOSCOW_PRIORITY_LABELS = (
    "Must Have",
    "Should Have",
    "Could Have",
    "Won't Have",
)


def group_moscow_requirements(requirements):
    requirements = requirements or []
    return [
        {
            "label": label,
            "requirements": [
                requirement
                for requirement in requirements
                if requirement.get("moscow_priority") == label
            ],
        }
        for label in MOSCOW_PRIORITY_LABELS
    ]


def group_explicit_requirements(requirements):
    requirements = requirements or []
    return [
        {
            "label": priority_label,
            "requirements": [
                {
                    **requirement,
                    "feature_label": (
                        EXPLICIT_REQUIREMENT_FEATURE_LABELS[
                            requirement["feature_key"]
                        ]
                    ),
                }
                for requirement in requirements
                if requirement.get("priority") == priority
            ],
        }
        for priority, priority_label in (
            EXPLICIT_REQUIREMENT_PRIORITIES
        )
    ]

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
    search_scope = "all"
    base_model_family = None
    availability_summary = None
    candidate_results = []
    moscow_requirements = []
    explicit_requirements = []

    if request.method == "POST":
        request.session.pop("hugselect_comparison", None)
        request.session.pop("hugselect_decision_stress", None)
        query = request.POST.get("query", "").strip()
        try:
            explicit_requirements = parse_explicit_requirements(
                request.POST.getlist("requirement_feature"),
                request.POST.getlist("requirement_value"),
                request.POST.getlist("requirement_priority"),
            )
        except ValueError as exc:
            error = str(exc)

        requested_scope = request.POST.get(
            "search_scope",
            "all",
        ).strip()

        if requested_scope == "family":
            search_scope = "family"
            base_model_family = normalize_base_model_family(
                request.POST.get("base_model_family")
            )

            if base_model_family is None and error is None:
                error = (
                    "Please select a supported base-model family."
                )

        if query and error is None:
            try:
                search_kwargs = {
                    "limit": AVAILABILITY_CANDIDATE_LIMIT,
                    "base_model_family": base_model_family,
                }
                if explicit_requirements:
                    search_kwargs["explicit_requirements"] = (
                        explicit_requirements
                    )
                candidate_results = search_models_feature_based(
                    query,
                    **search_kwargs,
                )
                search_mode = "feature-based"
                if candidate_results:
                    moscow_requirements = (
                        candidate_results[0].get(
                            "moscow_requirements"
                        )
                        or []
                    )

            except Exception:
                logger.exception(
                    "Feature-based search failed; trying basic fallback."
                )

                try:
                    candidate_results = search_models_basic(
                        query,
                        limit=AVAILABILITY_CANDIDATE_LIMIT,
                        base_model_family=base_model_family,
                    )
                    search_mode = "basic-fallback"
                    warning = (
                        "Advanced recommendation is temporarily unavailable. "
                        "Showing basic search results instead."
                    )
                    if explicit_requirements:
                        warning += (
                            " Explicit MoSCoW priorities are not applied "
                            "by the basic fallback."
                        )

                except Exception:
                    logger.exception(
                        "Basic fallback search also failed."
                    )
                    error = (
                        "Model search is temporarily unavailable. "
                        "Please check that Elasticsearch is running and try again."
                    )

            if search_mode and error is None and candidate_results:
                results, availability_summary = select_available_results(
                    candidate_results,
                    display_limit=DISPLAY_RESULT_LIMIT,
                )

                if len(results) < DISPLAY_RESULT_LIMIT:
                    model_label = (
                        "model" if len(results) == 1 else "models"
                    )
                    availability_warning = (
                        f"Only {len(results)} {model_label} could be verified "
                        "as currently available."
                    )
                    warning = " ".join(
                        message
                        for message in (
                            warning,
                            availability_warning,
                        )
                        if message
                    )
        else:
            if not query and error is None:
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
            "availability": {
                result["model_id"]: result.get("availability")
                for result in results
                if result.get("model_id") and result.get("availability")
            },
            "moscow_requirements": moscow_requirements,
            "explicit_requirements": explicit_requirements,
        }
    if request.method == "POST":
        request.session["hugselect_search_results"] = {
            "query": query,
            "results": results,
            "warning": warning,
            "error": error,
            "search_mode": search_mode,
            "search_scope": search_scope,
            "base_model_family": base_model_family,
            "availability_summary": availability_summary,
            "moscow_requirements": moscow_requirements,
            "explicit_requirements": explicit_requirements,
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
            "explicit_requirement_feature_groups": (
                EXPLICIT_REQUIREMENT_FEATURE_GROUPS
            ),
            "explicit_requirement_priorities": (
                EXPLICIT_REQUIREMENT_PRIORITIES
            ),
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
    search_scope = search_state.get(
        "search_scope",
        "all",
    )
    base_model_family = normalize_base_model_family(
        search_state.get("base_model_family")
    )
    moscow_requirements = search_state.get(
        "moscow_requirements",
        [],
    )
    explicit_requirements = search_state.get(
        "explicit_requirements",
        [],
    )

    if search_scope != "family" or base_model_family is None:
        search_scope = "all"
        base_model_family = None

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
            "search_scope": search_scope,
            "base_model_family": base_model_family,
            "availability_summary": search_state.get(
                "availability_summary"
            ),
            "moscow_requirements": moscow_requirements,
            "explicit_requirements": explicit_requirements,
            "explicit_requirement_groups": (
                group_explicit_requirements(
                    explicit_requirements
                )
            ),
            "moscow_requirement_groups": (
                group_moscow_requirements(
                    moscow_requirements
                )
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
    availability = last_search.get("availability", {})

    search_context = {
        "query": last_search.get("query"),
        "search_mode": last_search.get("search_mode"),
        "score": scores.get(model_id),
        "explanation": explanations.get(model_id),
        "availability": availability.get(model_id),
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

    request.session["hugselect_comparison"] = {
        "model_ids": [model["model_id"] for model in models],
        "models": [
            {
                key: model.get(key)
                for key in (
                    "model_id",
                    "author",
                    "pipeline_tag",
                    "license",
                    "library_name",
                    "basemodels",
                    "comparison_score",
                )
            }
            for model in models
        ],
        "coverage_rows": coverage_rows,
        "decision_summary": decision_summary,
        "search_mode": search_mode,
    }
    request.session.pop("hugselect_decision_stress", None)

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

    request.session["hugselect_decision_stress"] = {
        "model_ids": model_ids,
        "stress_result": stress_result,
        "stress_summary": stress_summary,
        "essential_requirements": essential_requirements,
        "missing_explanation_ids": missing_explanation_ids,
    }

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


def analysis_report_view(request):
    """Download the currently saved HugSelect analysis as a PDF."""
    pdf_bytes = generate_analysis_report(
        search_state=request.session.get(
            "hugselect_search_results",
            {},
        ),
        comparison_state=request.session.get(
            "hugselect_comparison",
            {},
        ),
        decision_stress_state=request.session.get(
            "hugselect_decision_stress",
            {},
        ),
        generated_at=timezone.localtime(),
    )
    response = HttpResponse(
        pdf_bytes,
        content_type="application/pdf",
    )
    response["Content-Disposition"] = (
        'attachment; filename="hugselect-analysis-report.pdf"'
    )
    return response
