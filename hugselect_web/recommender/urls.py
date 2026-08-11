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
        "compare/save/",
        views.save_comparison_view,
        name="save_comparison",
    ),
    path(
        "comparisons/saved/",
        views.saved_comparisons_view,
        name="saved_comparisons",
    ),
    path(
        "comparisons/saved/<str:saved_id>/open/",
        views.open_saved_comparison_view,
        name="open_saved_comparison",
    ),
    path(
        "comparisons/saved/<str:saved_id>/remove/",
        views.remove_saved_comparison_view,
        name="remove_saved_comparison",
    ),
    path(
        "compare/stress-test/",
        views.decision_stress_view,
        name="decision_stress",
    ),
]
