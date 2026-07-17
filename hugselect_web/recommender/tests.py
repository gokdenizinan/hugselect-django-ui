from django.test import SimpleTestCase
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from .services import build_model_graph


class BuildModelGraphTests(SimpleTestCase):
    def test_builds_expected_graph(self):
        model = {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "author": "Qwen",
            "pipeline_tag": "text-generation",
            "license": "apache-2.0",
            "library_name": "transformers",
            "language": ["English"],
            "basemodels": ["Qwen/Qwen2.5-7B"],
            "model_type": "qwen2",
        }

        graph = build_model_graph(model)
        nodes_by_id = {
            node["id"]: node
            for node in graph["nodes"]
        }

        self.assertEqual(
            nodes_by_id["model"]["label"],
            "Qwen/Qwen2.5-7B-Instruct",
        )
        self.assertEqual(
            nodes_by_id["base_model"]["label"],
            "Qwen/Qwen2.5-7B",
        )
        self.assertEqual(
            nodes_by_id["language_0"]["label"],
            "English",
        )
        self.assertIn(
            {
                "source": "model",
                "target": "author",
                "label": "Created by",
            },
            graph["edges"],
        )
class CompareModelsViewTests(TestCase):
    def test_requires_two_or_three_models(self):
        response = self.client.get(
            reverse("compare_models"),
            {"model_ids": ["model-a"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Please select two or three models to compare.",
        )

    @patch("recommender.views.get_model_by_id")
    def test_builds_feature_comparison_context(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "language": ["English"],
                "basemodels": ["base/model"],
            },
            {
                "model_id": "author/model-b",
                "language": ["English", "Turkish"],
                "basemodels": [],
            },
        ]

        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "English text-generation model",
            "search_mode": "feature-based",
            "scores": {
                "author/model-a": 90.0,
                "author/model-b": 80.0,
            },
            "explanations": {
                "author/model-a": {
                    "per_feature": [
                        {
                            "matches": [
                                {
                                    "feature_key": "language",
                                    "user_value": "English",
                                    "matched": True,
                                }
                            ]
                        }
                    ]
                },
                "author/model-b": {
                    "per_feature": [
                        {
                            "matches": [
                                {
                                    "feature_key": "language",
                                    "user_value": "English",
                                    "matched": False,
                                }
                            ]
                        }
                    ]
                },
            },
        }
        session.save()

        response = self.client.get(
            reverse("compare_models"),
            {
                "model_ids": [
                    "author/model-a",
                    "author/model-b",
                ]
            },
        )

        self.assertEqual(response.status_code, 200)

        models = response.context["models"]
        coverage_rows = response.context["coverage_rows"]

        self.assertTrue(models[0]["is_strongest_match"])
        self.assertFalse(models[1]["is_strongest_match"])
        self.assertEqual(models[0]["language"], "English")
        self.assertEqual(
            models[1]["language"],
            "English, Turkish",
        )
        self.assertEqual(
            coverage_rows[0]["model_statuses"],
            [
                {
                    "model_id": "author/model-a",
                    "matched": True,
                },
                {
                    "model_id": "author/model-b",
                    "matched": False,
                },
            ],
        )
class DecisionStressViewTests(TestCase):
    def test_requires_two_or_three_models(self):
        response = self.client.get(
            reverse("decision_stress"),
            {"model_ids": ["author/model-a"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            (
                "Please select two or three models "
                "to run a decision stress test."
            ),
        )

    def test_displays_selected_models_and_search_query(self):
        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "English text-generation model",
            "search_mode": "feature-based",
        }
        session.save()

        response = self.client.get(
            reverse("decision_stress"),
            {
                "model_ids": [
                    "author/model-a",
                    "author/model-b",
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "author/model-a")
        self.assertContains(response, "author/model-b")
        self.assertContains(
            response,
            "English text-generation model",
        )