from django.urls import path
from . import views

urlpatterns = [
    path("search/", views.search_view, name="search"),
    path(
        "results/",
        views.search_results_view,
        name="search_results",
    ),
    path(
        "results/report.pdf",
        views.analysis_report_view,
        name="analysis_report",
    ),
    path(
        "models/<path:model_id>/",
        views.model_detail_view,
        name="model_detail",
    ),
    path("compare/", views.compare_models_view, name="compare_models"),
    path(
    "compare/stress-test/",
    views.decision_stress_view,
    name="decision_stress",
    ),
    ]
