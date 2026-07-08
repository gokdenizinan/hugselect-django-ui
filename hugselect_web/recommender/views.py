from django.shortcuts import render
from .services import search_models_basic


def search_view(request):
    query = ""
    results = []
    error = None

    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            try:
                results = search_models_basic(query, limit=10)
            except Exception as exc:
                error = str(exc)

    return render(request, "recommender/search.html", {
        "query": query,
        "results": results,
        "error": error,
    })