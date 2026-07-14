from django.urls import path
from . import views

urlpatterns = [
    path("search/", views.search_view, name="search"),
    path(
        "models/<path:model_id>/",
        views.model_detail_view,
        name="model_detail",
    ),
    ]