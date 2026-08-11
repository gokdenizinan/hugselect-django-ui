import logging
import uuid
from urllib.parse import urlencode

from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import Http404, HttpResponse, HttpResponseBadRequest
from django.urls import reverse
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views.decorators.http import require_GET, require_POST
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
SAVED_COMPARISONS_SESSION_KEY = "hugselect_saved_comparisons"
MAX_SAVED_COMPARISONS = 10


def _saved_comparisons(request):
    saved_comparisons = request.session.get(
        SAVED_COMPARISONS_SESSION_KEY,
        [],
    )
    return saved_comparisons if isinstance(saved_comparisons, list) else []


def _saved_comparison_or_404(request, saved_id):
    for comparison in _saved_comparisons(request):
        if comparison.get("id") == saved_id:
            return comparison
    raise Http404("Saved comparison not found.")


def _comparison_search_context(search_state, model_ids):
    search_state = search_state or {}
    scores = search_state.get("scores", {}) or {}
    explanations = search_state.get("explanations", {}) or {}
    return {
        "query": search_state.get("query"),
        "search_mode": search_state.get("search_mode"),
        "scores": {
            model_id: scores[model_id]
            for model_id in model_ids
            if model_id in scores
        },
        "explanations": {
            model_id: explanations[model_id]
            for model_id in model_ids
            if model_id in explanations
        },
    }


def _generated_comparison_label(model_ids):
    model_names = [
        model_id.rsplit("/", 1)[-1]
        for model_id in model_ids
    ]
    return f"Comparison — {' vs '.join(model_names)}"


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
            "saved_comparison_count": len(
                _saved_comparisons(request)
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
    saved_id = request.GET.get("saved_comparison")
    saved_comparison = None

    if saved_id:
        saved_comparison = _saved_comparison_or_404(
            request,
            saved_id,
        )
        model_ids = saved_comparison.get("model_ids", [])
    else:
        model_ids = request.GET.getlist("model_ids")

    model_ids = list(dict.fromkeys(model_ids))
    saved_comparison_count = len(_saved_comparisons(request))

    if len(model_ids) < 2 or len(model_ids) > 3:
        return render(
            request,
            "recommender/compare.html",
            {
                "error": "Please select two or three models to compare.",
                "models": [],
                "saved_comparison_count": saved_comparison_count,
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
                "saved_comparison_count": saved_comparison_count,
            },
        )

    if saved_comparison is not None:
        last_search = saved_comparison.get("search_context", {}) or {}
    else:
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
        "search_context": _comparison_search_context(
            last_search,
            [model["model_id"] for model in models],
        ),
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
            "saved_comparison": saved_comparison,
            "saved_context_unavailable": (
                saved_comparison is not None
                and not any((
                    last_search.get("query"),
                    last_search.get("search_mode"),
                    last_search.get("scores"),
                    last_search.get("explanations"),
                ))
            ),
            "saved_comparison_count": saved_comparison_count,
        },
    )


@require_POST
def save_comparison_view(request):
    model_ids = list(
        dict.fromkeys(request.POST.getlist("model_ids"))
    )
    current_comparison = request.session.get(
        "hugselect_comparison",
        {},
    )
    current_model_ids = current_comparison.get("model_ids", [])

    if (
        len(model_ids) < 2
        or len(model_ids) > 3
        or model_ids != current_model_ids
    ):
        return HttpResponseBadRequest(
            "Only the current two- or three-model comparison can be saved."
        )

    search_context = current_comparison.get("search_context")
    if not search_context:
        search_context = _comparison_search_context(
            request.session.get("hugselect_last_search", {}),
            model_ids,
        )

    saved_comparisons = list(_saved_comparisons(request))
    duplicate = next(
        (
            comparison
            for comparison in saved_comparisons
            if set(comparison.get("model_ids", [])) == set(model_ids)
            and (
                comparison.get("search_context", {}).get("query")
                == search_context.get("query")
            )
            and (
                comparison.get("search_context", {}).get("search_mode")
                == search_context.get("search_mode")
            )
        ),
        None,
    )
    if duplicate is not None:
        messages.info(
            request,
            "This comparison is already saved for the same search context.",
        )
        return redirect("saved_comparisons")

    if len(saved_comparisons) >= MAX_SAVED_COMPARISONS:
        messages.error(
            request,
            (
                f"You can save up to {MAX_SAVED_COMPARISONS} comparisons "
                "in this session. Remove one before saving another."
            ),
        )
        return redirect("saved_comparisons")

    requested_label = request.POST.get("label", "").strip()
    label = (
        requested_label[:100]
        if requested_label
        else _generated_comparison_label(model_ids)
    )
    saved_comparisons.append({
        "id": uuid.uuid4().hex,
        "label": label,
        "model_ids": model_ids,
        "saved_at": timezone.now().isoformat(),
        "search_context": search_context,
    })
    request.session[SAVED_COMPARISONS_SESSION_KEY] = saved_comparisons
    messages.success(request, "Comparison saved for this session.")
    return redirect("saved_comparisons")


@require_GET
def saved_comparisons_view(request):
    saved_comparisons = []
    for comparison in _saved_comparisons(request):
        saved_at = parse_datetime(comparison.get("saved_at") or "")
        if saved_at is not None:
            if timezone.is_naive(saved_at):
                saved_at = timezone.make_aware(saved_at)
            saved_at_display = timezone.localtime(saved_at).strftime(
                "%d %b %Y, %H:%M"
            )
        else:
            saved_at_display = "Time unavailable"
        saved_comparisons.append({
            **comparison,
            "saved_at_display": saved_at_display,
        })

    return render(
        request,
        "recommender/saved_comparisons.html",
        {
            "saved_comparisons": saved_comparisons,
            "maximum_saved_comparisons": MAX_SAVED_COMPARISONS,
        },
    )


@require_GET
def open_saved_comparison_view(request, saved_id):
    _saved_comparison_or_404(request, saved_id)
    query_string = urlencode({"saved_comparison": saved_id})
    return redirect(f"{reverse('compare_models')}?{query_string}")


@require_POST
def remove_saved_comparison_view(request, saved_id):
    saved_comparison = _saved_comparison_or_404(request, saved_id)
    request.session[SAVED_COMPARISONS_SESSION_KEY] = [
        comparison
        for comparison in _saved_comparisons(request)
        if comparison.get("id") != saved_comparison.get("id")
    ]
    messages.success(request, "Saved comparison removed.")
    return redirect("saved_comparisons")


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
