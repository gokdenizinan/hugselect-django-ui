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