import logging

from django.shortcuts import render

from .services import (
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
