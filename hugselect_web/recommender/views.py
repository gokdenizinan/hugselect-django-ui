from django.shortcuts import render

from .services import (
    search_models_basic,
    search_models_feature_based,
)


def search_view(request):
    query = ""
    results = []
    error = None
    search_mode = None

    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            try:
                results = search_models_feature_based(query, limit=10)
                search_mode = "feature-based"

            except Exception as feature_error:
                try:
                    results = search_models_basic(query, limit=10)
                    search_mode = "basic-fallback"
                    error = (
                        "Feature-based search was unavailable, so basic search "
                        f"was used instead. Reason: {feature_error}"
                    )

                except Exception as basic_error:
                    error = (
                        "Both feature-based and basic search failed. "
                        f"Feature error: {feature_error}. "
                        f"Basic error: {basic_error}"
                    )

    return render(
        request,
        "recommender/search.html",
        {
            "query": query,
            "results": results,
            "error": error,
            "search_mode": search_mode,
        },
    )
