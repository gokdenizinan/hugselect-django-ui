from django.test import SimpleTestCase
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch
from django.test import TestCase
from django.urls import reverse
from .services import (
    AVAILABILITY_CANDIDATE_LIMIT,
    DISPLAY_RESULT_LIMIT,
    EXPLICIT_REQUIREMENT_FEATURE_GROUPS,
    EXPLICIT_REQUIREMENT_PRIORITIES,
    _base_model_family_filter,
    _make_feature_search_builder,
    build_model_graph,
    check_model_availability,
    parse_explicit_requirements,
    search_models_basic,
    search_models_feature_based,
    select_available_results,
)
from .decision_stress import (
    SCENARIOS,
    category_for_feature,
    explain_scenario_outcome,
    run_decision_stress_test,
    score_explanation_for_scenario,
    summarize_decision_stress_test,

)
from .reporting import (
    build_analysis_report_data,
    generate_analysis_report,
)
from EE_Query_Builder_Clean_modified_v3_dedupfix import (
    EXPLICIT_MOSCOW_PRIORITIES,
    FeatureGroup,
    PRIORITY_TO_MOSCOW,
)


def _select_mock_results_as_available(
    ranked_results,
    display_limit=DISPLAY_RESULT_LIMIT,
):
    selected_results = list(ranked_results[:display_limit])

    return selected_results, {
        "candidate_count": len(ranked_results),
        "checked_count": len(selected_results),
        "http_check_count": len(selected_results),
        "available_count": len(selected_results),
        "unavailable_count": 0,
        "unknown_count": 0,
        "display_limit": display_limit,
        "shortfall": max(0, display_limit - len(selected_results)),
    }


class MockAvailabilitySelectionMixin:
    def setUp(self):
        super().setUp()
        patcher = patch(
            "recommender.views.select_available_results",
            side_effect=_select_mock_results_as_available,
        )
        self.addCleanup(patcher.stop)
        self.mock_select_available_results = patcher.start()

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
class SearchResultGroupingTests(SimpleTestCase):
    def test_groups_results_by_base_model_count(self):
        from .views import group_search_results_by_base_model

        results = [
            {
                "model_id": "author/model-a",
                "basemodels": "base/shared",
            },
            {
                "model_id": "author/model-b",
                "basemodels": ["base/shared"],
            },
            {
                "model_id": "author/model-c",
                "basemodels": [
                    "base/first",
                    "base/second",
                ],
            },
            {
                "model_id": "author/model-d",
                "basemodels": None,
            },
        ]

        grouped_results = group_search_results_by_base_model(
            results
        )
        self.assertEqual(
            [
                result["tooltip_key"]
                for result in results
            ],
            [
                "result-1",
                "result-2",
                "result-3",
                "result-4",
            ],
        )

        self.assertEqual(
            grouped_results["single_base_model_groups"],
            [
                {
                    "base_model": "base/shared",
                    "results": [
                        results[0],
                        results[1],
                    ],
                }
            ],
        )

        self.assertEqual(
            grouped_results["multiple_base_model_results"],
            [results[2]],
        )

        self.assertEqual(
            grouped_results["no_base_model_results"],
            [results[3]],
        )
class SearchViewGroupingTests(
    MockAvailabilitySelectionMixin,
    TestCase,
):
    @patch("recommender.views.search_models_feature_based")
    def test_adds_grouped_results_to_search_context(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "basemodels": "base/shared",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "basemodels": ["base/shared"],
                "score": 80.0,
            },
            {
                "model_id": "author/model-c",
                "basemodels": [
                    "base/first",
                    "base/second",
                ],
                "score": 70.0,
            },
            {
                "model_id": "author/model-d",
                "basemodels": None,
                "score": 60.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {"query": "English text-generation model"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)

        grouped_results = response.context["grouped_results"]

        self.assertEqual(
            grouped_results["single_base_model_groups"][0][
                "base_model"
            ],
            "base/shared",
        )
        self.assertEqual(
            len(
                grouped_results[
                    "single_base_model_groups"
                ][0]["results"]
            ),
            2,
        )
        self.assertEqual(
            grouped_results[
                "multiple_base_model_results"
            ][0]["model_id"],
            "author/model-c",
        )
        self.assertEqual(
            grouped_results[
                "no_base_model_results"
            ][0]["model_id"],
            "author/model-d",
        )
class SearchTemplateGroupingTests(
    MockAvailabilitySelectionMixin,
    TestCase,
):
    @patch("recommender.views.search_models_feature_based")
    def test_hides_empty_base_model_categories(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "basemodels": "base/shared",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "basemodels": ["base/shared"],
                "score": 80.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {"query": "English text-generation model"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Single base model",
        )
        self.assertNotContains(
            response,
            "Multiple base models",
        )
        self.assertNotContains(
            response,
            "No base model",
        )
class SearchViewTooltipTests(
    MockAvailabilitySelectionMixin,
    TestCase,
):
    @patch("recommender.views.search_models_feature_based")
    def test_explains_task_metadata_for_each_search_result(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "author": "Author A",
                "pipeline_tag": "text-generation",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "author": "Author B",
                "pipeline_tag": "summarization",
                "score": 80.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {
                "query": "English text-generation model",
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        response_html = response.content.decode()
        result_cards = response_html.split(
            '<article class="result-card">'
        )[1:]

        self.assertEqual(len(result_cards), 2)
        self.assertRegex(result_cards[0], r"Author:\s+Author A")
        self.assertRegex(result_cards[1], r"Author:\s+Author B")
        self.assertRegex(result_cards[0], r"Task:\s+text-generation")
        self.assertRegex(result_cards[1], r"Task:\s+summarization")
        self.assertEqual(result_cards[0].count("Task:"), 1)
        self.assertEqual(result_cards[1].count("Task:"), 1)
        self.assertIn(
            'aria-describedby="search-task-tooltip-result-1"',
            result_cards[0],
        )
        self.assertIn(
            'id="search-task-tooltip-result-1"',
            result_cards[0],
        )
        self.assertIn(
            'aria-describedby="search-task-tooltip-result-2"',
            result_cards[1],
        )
        self.assertIn(
            'id="search-task-tooltip-result-2"',
            result_cards[1],
        )

    @patch("recommender.views.search_models_feature_based")
    def test_explains_library_metadata_for_each_search_result(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "library_name": "transformers",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "library_name": "diffusers",
                "score": 80.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {"query": "English text-generation model"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "transformers")
        self.assertContains(response, "diffusers")

        response_html = response.content.decode()
        result_cards = response_html.split(
            '<article class="result-card">'
        )[1:]

        self.assertEqual(len(result_cards), 2)
        self.assertIn(
            'aria-describedby="search-library-tooltip-result-1"',
            result_cards[0],
        )
        self.assertIn(
            'id="search-library-tooltip-result-1"',
            result_cards[0],
        )
        self.assertRegex(
            result_cards[0],
            (
                r"Library:\s+transformers[\s\S]*"
                r"“transformers” refers to Hugging Face’s\s+"
                r"library for working with pretrained models\."
            ),
        )
        self.assertIn(
            'aria-describedby="search-library-tooltip-result-2"',
            result_cards[1],
        )
        self.assertIn(
            'id="search-library-tooltip-result-2"',
            result_cards[1],
        )
        self.assertRegex(result_cards[1], r"Library:\s+diffusers")
        self.assertNotIn("“transformers” refers to", result_cards[1])
        self.assertRegex(
            response_html,
            (
                r"The software library used to load, run,\s+"
                r"or fine-tune the model\."
            ),
        )

    @patch("recommender.views.search_models_feature_based")
    def test_explains_license_metadata_for_each_search_result(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "license": "apache-2.0",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "license": "bsd-3-clause",
                "score": 80.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {"query": "English text-generation model"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "apache-2.0")
        self.assertContains(response, "bsd-3-clause")
        self.assertContains(
            response,
            'aria-describedby="search-license-tooltip-result-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="search-license-tooltip-result-2"',
        )
        self.assertContains(response, 'id="search-license-tooltip-result-1"')
        self.assertContains(response, 'id="search-license-tooltip-result-2"')

        response_html = response.content.decode()
        result_cards = response_html.split(
            '<article class="result-card">'
        )[1:]

        self.assertRegex(result_cards[0], r"License:\s+apache-2\.0")
        self.assertRegex(result_cards[1], r"License:\s+bsd-3-clause")
        self.assertRegex(
            response_html,
            (
                r"The license describes the rules for using,\s+"
                r"modifying, and redistributing the model\.\s+"
                r"Exact permissions and obligations depend on\s+"
                r"the specific license shown\."
            ),
        )

    @patch("recommender.views.search_models_feature_based")
    def test_explains_feature_match_for_each_search_result(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": "author/model-a",
                "score": 90.0,
            },
            {
                "model_id": "author/model-b",
                "score": 80.0,
            },
        ]

        response = self.client.post(
            reverse("search"),
            {"query": "English text-generation model"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "90.0%")
        self.assertContains(response, "80.0%")
        self.assertContains(
            response,
            'aria-describedby="search-feature-match-tooltip-result-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="search-feature-match-tooltip-result-2"',
        )
        self.assertContains(
            response,
            'id="search-feature-match-tooltip-result-1"',
        )
        self.assertContains(
            response,
            'id="search-feature-match-tooltip-result-2"',
        )

        response_html = response.content.decode()
        self.assertRegex(
            response_html,
            (
                r"Feature match shows how much of your weighted\s+"
                r"requirements this model satisfies\. It does not\s+"
                r"measure general model quality or benchmark performance\."
            ),
        )

class ModelDetailViewTests(TestCase):
    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")
    def test_explains_library_metadata(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-a",
            "library_name": "transformers",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-a"],
            )
        )

        self.assertEqual(response.status_code, 200)
        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"The software library used to load, run,\s+"
                r"or fine-tune the model\."
            ),
        )

        self.assertRegex(
            response_html,
        (
            r"“transformers” refers to\s+"
            r"Hugging Face’s\s+"
            r"library for working with pretrained models\."
        ),
    )

        self.assertContains(
            response,
            'aria-describedby="library-tooltip"',
        )
    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")

    def test_does_not_describe_other_libraries_as_transformers(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-b",
            "library_name": "diffusers",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-b"],
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "diffusers",
        )
        self.assertNotContains(
            response,
            "“transformers” refers to",
        )
    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")
    def test_explains_task_metadata(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-c",
            "pipeline_tag": "text-generation",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-c"],
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "text-generation",
        )

        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"The task describes the model's primary capability\s+"
                r"on Hugging Face,\s+"
                r"such as text generation,\s+"
                r"summarization, translation,\s+"
                r"or question answering\."
            ),
        )

        self.assertContains(
            response,
            'aria-describedby="task-tooltip"',
        )
    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")
    def test_explains_license_metadata(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-d",
            "license": "apache-2.0",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-d"],
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "apache-2.0",
        )

        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"The license describes the rules for using,\s+"
                r"modifying, and redistributing the model\.\s+"
                r"Exact permissions and obligations depend on\s+"
                r"the specific license shown\."
            ),
        )

        self.assertContains(
            response,
            'aria-describedby="license-tooltip"',
        )

    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")
    def test_links_back_to_search_results(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-a",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-a"],
            )
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'href="{reverse("search_results")}"',
            count=1,
        )
        self.assertNotContains(response, "window.history.back()")

    @patch("recommender.views.build_model_graph")
    @patch("recommender.views.get_model_by_id")
    def test_reuses_stored_availability_on_model_detail(
        self,
        mock_get_model_by_id,
        mock_build_model_graph,
    ):
        mock_get_model_by_id.return_value = {
            "model_id": "author/model-a",
        }
        mock_build_model_graph.return_value = {
            "nodes": [],
            "edges": [],
        }
        session = self.client.session
        session["hugselect_last_search"] = {
            "availability": {
                "author/model-a": {
                    "status": "available",
                    "http_status": 200,
                }
            }
        }
        session.save()

        response = self.client.get(
            reverse(
                "model_detail",
                args=["author/model-a"],
            )
        )

        self.assertEqual(
            response.context["search_context"]["availability"],
            {"status": "available", "http_status": 200},
        )
        self.assertContains(response, "Available")

class CompareModelsViewTests(TestCase):
    def test_links_back_to_search_results(self):
        response = self.client.get(
            reverse("compare_models"),
            {"model_ids": ["model-a"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'href="{reverse("search_results")}"',
            count=1,
        )

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
    def test_explains_library_metadata_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "library_name": "transformers",
            },
            {
                "model_id": "author/model-b",
                "library_name": "diffusers",
            },
        ]

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
        self.assertContains(
            response,
            'aria-describedby="comparison-library-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-library-tooltip-2"',
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
        comparison_state = self.client.session[
            "hugselect_comparison"
        ]
        self.assertEqual(
            comparison_state["model_ids"],
            ["author/model-a", "author/model-b"],
        )
        self.assertEqual(
            comparison_state["coverage_rows"],
            coverage_rows,
        )
    @patch("recommender.views.get_model_by_id")
    def test_explains_task_metadata_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "pipeline_tag": "text-generation",
            },
            {
                "model_id": "author/model-b",
                "pipeline_tag": "summarization",
            },
        ]

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
        self.assertContains(
            response,
            "text-generation",
        )
        self.assertContains(
            response,
            "summarization",
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-task-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-task-tooltip-2"',
        )
    @patch("recommender.views.get_model_by_id")
    def test_explains_license_metadata_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "license": "apache-2.0",
            },
            {
                "model_id": "author/model-b",
                "license": "mit",
            },
        ]

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
        self.assertContains(response, "apache-2.0")
        self.assertContains(response, "mit")
        self.assertContains(
            response,
            'aria-describedby="comparison-license-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-license-tooltip-2"',
        )
    @patch("recommender.views.get_model_by_id")
    def test_explains_base_model_metadata_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "basemodels": ["base/model-a"],
            },
            {
                "model_id": "author/model-b",
                "basemodels": ["base/model-b"],
            },
        ]

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
        self.assertContains(response, "base/model-a")
        self.assertContains(response, "base/model-b")

        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"The base model is the model used as the "
                r"starting point\s+"
                r"before additional training or fine-tuning\."
            ),
        )

        self.assertContains(
            response,
            'aria-describedby="comparison-base-model-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-base-model-tooltip-2"',
        )
    @patch("recommender.views.get_model_by_id")
    def test_explains_model_type_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
                "model_type": "qwen2",
            },
            {
                "model_id": "author/model-b",
                "model_type": "bert",
            },
        ]

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
        self.assertContains(response, "qwen2")
        self.assertContains(response, "bert")

        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"The model type identifies the model's\s+"
                r"architecture\s+or configuration family,\s+"
                r"such as BERT, Llama, or Qwen\."
            ),
        )

        self.assertContains(
            response,
            'aria-describedby="comparison-model-type-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-model-type-tooltip-2"',
        )
    @patch("recommender.views.get_model_by_id")
    def test_explains_feature_match_for_each_compared_model(
        self,
        mock_get_model_by_id,
    ):
        mock_get_model_by_id.side_effect = [
            {
                "model_id": "author/model-a",
            },
            {
                "model_id": "author/model-b",
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
            "explanations": {},
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
        self.assertContains(response, "90.0%")
        self.assertContains(response, "80.0%")

        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r"Feature match shows how much of your weighted\s+"
                r"requirements this model satisfies\.\s+"
                r"It does not measure general model quality\s+"
                r"or benchmark performance\."
            ),
        )

        self.assertContains(
            response,
            'aria-describedby="comparison-feature-match-tooltip-1"',
        )
        self.assertContains(
            response,
            'aria-describedby="comparison-feature-match-tooltip-2"',
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
        model_a_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                }
            ]
        }

        model_b_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                }
            ]
        }

        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "English text-generation model",
            "search_mode": "feature-based",
            "explanations": {
                "author/model-a": model_a_explanation,
                "author/model-b": model_b_explanation,
            },
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

        self.assertEqual(
            response.context["stress_summary"]["leaders"],
            ["author/model-a"],
        )
        stored_stress = self.client.session[
            "hugselect_decision_stress"
        ]
        self.assertEqual(
            stored_stress["stress_summary"]["leaders"],
            ["author/model-a"],
        )
        self.assertEqual(
            stored_stress["model_ids"],
            ["author/model-a", "author/model-b"],
        )
        self.assertContains(
            response,
            "Why excluded",
        )
        self.assertContains(
            response,
            "Task: text generation",)
        self.assertNotContains(
            response,
            "Why these models remain tied",
        )
        self.assertContains(
            response,
            "Overall",
        )
        self.assertNotContains(
            response,
            "Scenario score",
        )

        response_html = response.content.decode()
        tooltip_types = (
            "overall",
            "essential",
            "preference",
            "functional",
            "quality",
        )

        for scenario_number in range(1, len(SCENARIOS) + 1):
            for tooltip_type in tooltip_types:
                tooltip_id = (
                    f"stress-{tooltip_type}-tooltip-"
                    f"{scenario_number}"
                )
                self.assertContains(
                    response,
                    f'aria-describedby="{tooltip_id}"',
                    count=1,
                )
                self.assertContains(
                    response,
                    f'id="{tooltip_id}"',
                    count=1,
                )

        self.assertRegex(
            response_html,
            (
                r"Overall is the model's normalized weighted "
                r"requirement-match\s+"
                r"score under this priority scenario, calculated from stored\s+"
                r"match evidence\. It is not a benchmark or general "
                r"model-quality score\."
            ),
        )
        self.assertContains(
            response,
            'aria-describedby="outright-win-consistency-tooltip"',
        )
        self.assertContains(
            response,
            'id="outright-win-consistency-tooltip"',
        )
        self.assertContains(
            response,
            'aria-describedby="strict-essentials-tooltip"',
        )
        self.assertContains(
            response,
            'id="strict-essentials-tooltip"',
        )

    def test_displays_missing_stored_evidence(self):
        model_a_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                },
                {
                    "effective_weight": 8.0,
                    "matches": [
                        {
                            "feature_key": "license_name",
                            "user_value": "apache-2.0",
                            "effective_weight": 8.0,
                            "matched": True,
                            "score": 8.0,
                        }
                    ],
                },
            ]
        }

        model_b_explanation = {
            "per_feature": [
                {
                    "effective_weight": 8.0,
                    "matches": [
                        {
                            "feature_key": "license_name",
                            "user_value": "apache-2.0",
                            "effective_weight": 8.0,
                            "matched": True,
                            "score": 8.0,
                        }
                    ],
                }
            ]
        }

        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "Text generation with Apache license",
            "search_mode": "feature-based",
            "explanations": {
                "author/model-a": model_a_explanation,
                "author/model-b": model_b_explanation,
            },
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
        self.assertContains(
            response,
            "Missing stored evidence",
        )
        response_html = response.content.decode()

        normalized_html = " ".join(response_html.split())

        self.assertIn(
            (
                "No stored match record was found for one or more listed "
                "requirements. This does not confirm that the model lacks "
                "them, but HugSelect cannot count them as satisfied in "
                "this analysis."
            ),
            normalized_html,
        )

        self.assertNotContains(
            response,
            "missing-stored-evidence-tooltip",
        )

    def test_explains_evidence_equivalent_models(self):
        identical_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                }
            ]
        }

        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "English text-generation model",
            "search_mode": "feature-based",
            "explanations": {
                "author/model-a": identical_explanation,
                "author/model-b": identical_explanation,
            },
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
        self.assertTrue(
            response.context[
                "stress_summary"
            ]["all_scenarios_tied"]
        )
        self.assertContains(
            response,
            "5 / 5",
        )
        self.assertContains(
            response,
            "Tied scenarios",
        )
        self.assertNotContains(
            response,
            "Scenario consistency",
        )
        self.assertContains(
            response,
            "Why these models remain tied",
        )
        self.assertContains(
            response,
            (
                "The indexed requirement evidence "
                "cannot distinguish these models."
            ),
        )
        self.assertContains(
            response,
            "benchmark performance",
        )
    def test_displays_no_eligible_model_when_all_are_excluded(self):
        missing_essential_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                }
            ]
        }

        session = self.client.session
        session["hugselect_last_search"] = {
            "query": "I need a text-generation model",
            "search_mode": "feature-based",
            "explanations": {
                "author/model-a": missing_essential_explanation,
                "author/model-b": missing_essential_explanation,
            },
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

        strict_essentials = next(
            scenario
            for scenario in response.context[
                "stress_result"
            ]["scenarios"]
            if scenario["key"] == "strict_essentials"
        )

        self.assertTrue(
            strict_essentials["all_models_excluded"]
        )
        self.assertEqual(
            strict_essentials["winners"],
            [],
        )
        self.assertContains(
            response,
            "No eligible model",
        )

class DecisionStressPolicyTests(SimpleTestCase):
    def test_defines_five_unique_scenarios(self):
        scenario_keys = [
            scenario["key"]
            for scenario in SCENARIOS
        ]

        self.assertEqual(len(SCENARIOS), 5)
        self.assertEqual(
            len(scenario_keys),
            len(set(scenario_keys)),
        )

    def test_maps_features_to_expected_categories(self):
        self.assertEqual(
            category_for_feature("task"),
            "essential",
        )
        self.assertEqual(
            category_for_feature("license_name"),
            "preference",
        )
        self.assertEqual(
            category_for_feature("functional"),
            "functional",
        )
        self.assertEqual(
            category_for_feature("Reliability"),
            "quality",
        )
        self.assertEqual(
            category_for_feature("unrecognized_feature"),
            "unknown",
        )
    def test_names_current_scenario_as_evidence_baseline(self):
        current_scenario = next(
            scenario
            for scenario in SCENARIOS
            if scenario["key"] == "current"
        )

        self.assertEqual(
            current_scenario["label"],
            "Starting priorities",
        )
        self.assertEqual(
            current_scenario["description"],
            (
                "Uses the requirements from your search with "
                "the importance HugSelect originally assigned to them."
            ),
        )

class DecisionStressScoringTests(SimpleTestCase):
    def test_rescores_existing_match_evidence(self):
        explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                },
                {
                    "effective_weight": 8.0,
                    "matches": [
                        {
                            "feature_key": "license_name",
                            "user_value": "apache-2.0",
                            "effective_weight": 8.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                },
                {
                    "effective_weight": 12.0,
                    "matches": [
                        {
                            "feature_key": "functional",
                            "user_value": "answer questions",
                            "effective_weight": 12.0,
                            "matched": True,
                            "score": 12.0,
                        }
                    ],
                },
            ]
        }

        current_scenario = next(
            scenario
            for scenario in SCENARIOS
            if scenario["key"] == "current"
        )

        result = score_explanation_for_scenario(
            explanation,
            current_scenario,
        )

        self.assertEqual(result["raw_score"], 22.0)
        self.assertEqual(result["maximum_score"], 30.0)
        self.assertEqual(result["score"], 73.33)
        self.assertFalse(result["strict_exclusion"])
    def test_calculates_percentage_score_for_each_category(self):
        explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                },
                {
                    "effective_weight": 8.0,
                    "matches": [
                        {
                            "feature_key": "license_name",
                            "user_value": "apache-2.0",
                            "effective_weight": 8.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                },
                {
                    "effective_weight": 12.0,
                    "matches": [
                        {
                            "feature_key": "functional",
                            "user_value": "answer questions",
                            "effective_weight": 12.0,
                            "matched": True,
                            "score": 12.0,
                        }
                    ],
                },
            ]
        }

        current_scenario = next(
            scenario
            for scenario in SCENARIOS
            if scenario["key"] == "current"
        )

        result = score_explanation_for_scenario(
            explanation,
            current_scenario,
        )

        self.assertEqual(
            result["category_scores"],
            {
                "essential": 100.0,
                "preference": 0.0,
                "functional": 100.0,
                "quality": 0.0,
            },
        )

        self.assertEqual(
            result["category_maximums"],
            {
                "essential": 10.0,
                "preference": 8.0,
                "functional": 12.0,
                "quality": 0.0,
            },
        )
    def test_strict_essentials_excludes_missing_essential(self):
        explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                }
            ]
        }

        strict_scenario = next(
            scenario
            for scenario in SCENARIOS
            if scenario["key"] == "strict_essentials"
        )

        result = score_explanation_for_scenario(
            explanation,
            strict_scenario,
        )

        self.assertTrue(result["strict_exclusion"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(
            result["missed_essentials"],
            [
                {
                    "feature_key": "task",
                    "user_value": "text generation",
                }
            ],
        )

    def test_keeps_same_text_in_different_feature_types_separate(
        self,
    ):
        explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "objective",
                            "user_value": "apache-2.0",
                            "effective_weight": 10.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                },
                {
                    "effective_weight": 8.0,
                    "matches": [
                        {
                            "feature_key": "license_name",
                            "user_value": "apache-2.0",
                            "effective_weight": 8.0,
                            "matched": True,
                            "score": 8.0,
                        }
                    ],
                },
            ]
        }

        starting_scenario = next(
            scenario
            for scenario in SCENARIOS
            if scenario["key"] == "current"
        )

        result = score_explanation_for_scenario(
            explanation,
            starting_scenario,
        )

        self.assertEqual(result["raw_score"], 8.0)
        self.assertEqual(result["maximum_score"], 18.0)
        self.assertEqual(result["score"], 44.44)
    def test_deduplicates_true_feature_aliases(self):
        explanation = {
            "per_feature": [
                {
                    "effective_weight": 10,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text-generation",
                            "matched": True,
                            "score": 10,
                            "effective_weight": 10,
                        },
                        {
                            "feature_key": "task_alias",
                            "user_value": "text-generation",
                            "matched": True,
                            "score": 10,
                            "effective_weight": 10,
                        },
                    ],
                },
            ],
        }

        result = score_explanation_for_scenario(
            explanation,
            SCENARIOS[0],
        )

        self.assertEqual(result["raw_score"], 10)
        self.assertEqual(result["maximum_score"], 10)
        self.assertEqual(result["score"], 100)
class DecisionStressRankingTests(SimpleTestCase):
    def test_ranks_models_across_all_scenarios(self):
        model_explanations = {
            "author/model-a": {
                "per_feature": [
                    {
                        "effective_weight": 10.0,
                        "matches": [
                            {
                                "feature_key": "task",
                                "user_value": "text generation",
                                "effective_weight": 10.0,
                                "matched": True,
                                "score": 10.0,
                            }
                        ],
                    },
                    {
                        "effective_weight": 8.0,
                        "matches": [
                            {
                                "feature_key": "license_name",
                                "user_value": "apache-2.0",
                                "effective_weight": 8.0,
                                "matched": False,
                                "score": 0.0,
                            }
                        ],
                    },
                ]
            },
            "author/model-b": {
                "per_feature": [
                    {
                        "effective_weight": 10.0,
                        "matches": [
                            {
                                "feature_key": "task",
                                "user_value": "text generation",
                                "effective_weight": 10.0,
                                "matched": False,
                                "score": 0.0,
                            }
                        ],
                    },
                    {
                        "effective_weight": 8.0,
                        "matches": [
                            {
                                "feature_key": "license_name",
                                "user_value": "apache-2.0",
                                "effective_weight": 8.0,
                                "matched": True,
                                "score": 8.0,
                            }
                        ],
                    },
                ]
            },
        }

        result = run_decision_stress_test(
            model_explanations
        )

        self.assertEqual(result["scenario_count"], 5)
        self.assertEqual(len(result["scenarios"]), 5)

        current = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "current"
        )

        self.assertEqual(
            current["winners"],
            ["author/model-a"],
        )
        self.assertEqual(
            current["model_results"][0]["rank"],
            1,
        )

        preference_first = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "preference_first"
        )

        self.assertEqual(
            preference_first["winners"],
            ["author/model-b"],
        )
    def test_missing_shared_requirement_counts_against_model(
    self,
        ):
        model_explanations = {
            "author/model-a": {
                "per_feature": [
                    {
                        "effective_weight": 10.0,
                        "matches": [
                            {
                                "feature_key": "task",
                                "user_value": "text generation",
                                "effective_weight": 10.0,
                                "matched": True,
                                "score": 10.0,
                            }
                        ],
                    },
                    {
                        "effective_weight": 8.0,
                        "matches": [
                            {
                                "feature_key": "license_name",
                                "user_value": "apache-2.0",
                                "effective_weight": 8.0,
                                "matched": True,
                                "score": 8.0,
                            }
                        ],
                    },
                ]
            },
            "author/model-b": {
                "per_feature": [
                    {
                        "effective_weight": 8.0,
                        "matches": [
                            {
                                "feature_key": "license_name",
                                "user_value": "apache-2.0",
                                "effective_weight": 8.0,
                                "matched": True,
                                "score": 8.0,
                            }
                        ],
                    },
                ]
            },
        }

        result = run_decision_stress_test(
            model_explanations
        )

        starting_priorities = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "current"
        )

        starting_results_by_model = {
            model_result["model_id"]: model_result
            for model_result in starting_priorities[
                "model_results"
            ]
        }

        model_b_starting_result = (
            starting_results_by_model["author/model-b"]
        )

        self.assertEqual(
            model_b_starting_result["maximum_score"],
            18.0,
        )
        self.assertEqual(
            model_b_starting_result["score"],
            44.44,
        )
        self.assertEqual(
            model_b_starting_result["missing_requirements"],
            [
                {
                    "feature_key": "task",
                    "user_value": "text generation",
                }
            ],
        )

        strict_essentials = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "strict_essentials"
        )

        strict_results_by_model = {
            model_result["model_id"]: model_result
            for model_result in strict_essentials[
                "model_results"
            ]
        }

        self.assertTrue(
            strict_results_by_model[
                "author/model-b"
            ]["strict_exclusion"]
        )
    def test_preserves_joint_winners(self):
        identical_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": True,
                            "score": 10.0,
                        }
                    ],
                }
            ]
        }

        result = run_decision_stress_test({
            "author/model-a": identical_explanation,
            "author/model-b": identical_explanation,
        })

        current = result["scenarios"][0]

        self.assertTrue(current["is_tie"])
        self.assertEqual(
            current["winners"],
            [
                "author/model-a",
                "author/model-b",
            ],
        )
        self.assertEqual(
            [
                model["rank"]
                for model in current["model_results"]
            ],
            [1, 1],
        )
    def test_strict_essentials_has_no_winner_when_all_models_are_excluded(
        self,
    ):
        missing_essential_explanation = {
            "per_feature": [
                {
                    "effective_weight": 10.0,
                    "matches": [
                        {
                            "feature_key": "task",
                            "user_value": "text generation",
                            "effective_weight": 10.0,
                            "matched": False,
                            "score": 0.0,
                        }
                    ],
                }
            ]
        }

        result = run_decision_stress_test({
            "author/model-a": missing_essential_explanation,
            "author/model-b": missing_essential_explanation,
        })

        strict_essentials = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "strict_essentials"
        )

        self.assertEqual(
            strict_essentials["winners"],
            [],
        )
        self.assertTrue(
            strict_essentials["all_models_excluded"]
        )
    def test_uses_unrounded_scores_to_avoid_false_ties(self):
        def explanation_with_score(score):
            return {
                "per_feature": [
                    {
                        "effective_weight": 100.0,
                        "matches": [
                            {
                                "feature_key": "task",
                                "user_value": "text generation",
                                "effective_weight": 100.0,
                                "matched": True,
                                "score": score,
                            }
                        ],
                    }
                ]
            }

        result = run_decision_stress_test({
            "author/model-a": explanation_with_score(
                73.334
            ),
            "author/model-b": explanation_with_score(
                73.326
            ),
        })

        starting_priorities = next(
            scenario
            for scenario in result["scenarios"]
            if scenario["key"] == "current"
        )

        self.assertEqual(
            [
                model["score"]
                for model in starting_priorities["model_results"]
            ],
            [73.33, 73.33],
        )
        self.assertEqual(
            starting_priorities["winners"],
            ["author/model-a"],
        )
        self.assertFalse(
            starting_priorities["is_tie"]
        )

class DecisionStressSummaryTests(SimpleTestCase):
    def test_identifies_stable_winner(self):
        stress_result = {
            "scenarios": [
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-b"]},
            ]
        }

        summary = summarize_decision_stress_test(
            stress_result
        )

        self.assertEqual(
            summary["leaders"],
            ["author/model-a"],
        )
        self.assertEqual(
            summary["stability_percentage"],
            80.0,
        )
        self.assertEqual(
            summary["stability_label"],
            "Stable winner",
        )
        self.assertFalse(summary["is_tied"])

    def test_identifies_no_consistent_winner(self):
        stress_result = {
            "scenarios": [
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-b"]},
                {
                    "winners": [
                        "author/model-a",
                        "author/model-b",
                    ]
                },
                {"winners": ["author/model-a"]},
                {"winners": ["author/model-b"]},
            ]
        }

        summary = summarize_decision_stress_test(
            stress_result
        )

        self.assertEqual(
            summary["leaders"],
            [
                "author/model-a",
                "author/model-b",
            ],
        )
        self.assertEqual(
            summary["stability_label"],
            "No consistent winner",
        )
        self.assertTrue(summary["is_tied"])

    def test_identifies_evidence_equivalent_models(self):
        stress_result = {
            "scenarios": [
                {
                    "winners": [
                        "author/model-a",
                        "author/model-b",
                    ]
                }
                for _ in range(5)
            ]
        }

        summary = summarize_decision_stress_test(
            stress_result
        )

        self.assertEqual(
            summary["stability_label"],
            "Evidence-equivalent models",
        )
        self.assertEqual(
            summary["stability_percentage"],
            0.0,
        )
        self.assertEqual(
            summary["tied_scenario_count"],
            5,
        )
        self.assertTrue(
            summary["all_scenarios_tied"]
        )

    def test_does_not_count_no_eligible_winner_as_a_tie(self):
        stress_result = {
            "scenarios": [
                {
                    "winners": [
                        "author/model-a",
                        "author/model-b",
                    ],
                    "all_models_excluded": False,
                }
                for _ in range(4)
            ]
            + [
                {
                    "winners": [],
                    "all_models_excluded": True,
                }
            ]
        }

        summary = summarize_decision_stress_test(
            stress_result
        )

        self.assertEqual(
            summary["tied_scenario_count"],
            4,
        )
        self.assertEqual(
            summary["no_winner_scenario_count"],
            1,
        )
        self.assertFalse(
            summary["all_scenarios_tied"]
        )

class DecisionStressOutcomeExplanationTests(SimpleTestCase):
    def test_explains_single_winner(self):
        result = explain_scenario_outcome(
            model_results=[],
            winners=["model-a"],
            all_models_excluded=False,
        )

        self.assertEqual(result["kind"], "winner")
        self.assertEqual(result["title"], "Why model-a wins")
        self.assertEqual(result["summary"], "This model has the highest score under this scenario.")

    def test_explains_tied_winners(self):
        result = explain_scenario_outcome(
            model_results=[],
            winners=["model-a", "model-b"],
            all_models_excluded=False,
        )

        self.assertEqual(result["kind"], "tie")
        self.assertEqual(result["title"], "Tied result")
        self.assertEqual(result["summary"], "The leading models have the same scenario score.")

    def test_explains_when_all_models_are_excluded(self):
        result = explain_scenario_outcome(
            model_results=[],
            winners=[],
            all_models_excluded=True,
        )

        self.assertEqual(result["kind"], "no_eligible_model")
        self.assertEqual(result["title"], "No eligible model")
        self.assertEqual(result["summary"], "Every model missed at least one essential requirement.")

    def test_explains_when_no_result_is_available(self):
        result = explain_scenario_outcome(
            model_results=[],
            winners=[],
            all_models_excluded=False,
        )

        self.assertEqual(result["kind"], "no_result")
        self.assertEqual(result["title"], "No result")
        self.assertEqual(result["summary"], "No result")

    def test_explains_winners_strongest_category_advantage(self):
        model_results = [
            {
                "model_id": "model-a",
                "strict_exclusion": False,
                "category_scores": {
                    "essential": 100.0,
                    "preference": 50.0,
                    "functional": 80.0,
                    "quality": 0.0,
                },
                "category_maximums": {
                    "essential": 10.0,
                    "preference": 8.0,
                    "functional": 12.0,
                    "quality": 0.0,
                },
            },
            {
                "model_id": "model-b",
                "strict_exclusion": False,
                "category_scores": {
                    "essential": 80.0,
                    "preference": 60.0,
                    "functional": 70.0,
                    "quality": 0.0,
                },
                "category_maximums": {
                    "essential": 10.0,
                    "preference": 8.0,
                    "functional": 12.0,
                    "quality": 0.0,
                },
            },
        ]

        result = explain_scenario_outcome(
            model_results=model_results,
            winners=["model-a"],
            all_models_excluded=False,
        )

        self.assertEqual(
            result["summary"],
            (
                "Its clearest advantage over model-b is Essential "
                "(20.0 percentage points)."
            ),
        )
class SearchResultsViewTests(
    MockAvailabilitySelectionMixin,
    TestCase,
):
    def test_displays_separate_search_results_page(self):
        response = self.client.get(
            reverse("search_results")
        )
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(
            response,
            "recommender/search_results.html",
        )

    @patch("recommender.views.search_models_feature_based")
    def test_successful_search_redirects_and_stores_results(
        self,
        mock_search_models_feature_based,
    ):
        # Gerçek Gemini ve Elasticsearch çağrısı yerine
        # kontrollü bir arama sonucu döndürür.
        mock_results = [
            {
                "model_id": "author/model-a",
                "basemodels": ["base/model"],
                "score": 90.0,
                "moscow_requirements": [
                    {
                        "feature_key": "language",
                        "value": "English",
                        "priority": "must",
                        "moscow_priority": "Must Have",
                        "constraint_kind": "hard_positive",
                    }
                ],
            }
        ]
        mock_search_models_feature_based.return_value = mock_results

        # Kullanıcının arama formunu gönderme davranışını taklit eder.
        response = self.client.post(
            reverse("search"),
            {
                "query": "English text-generation model",
            },
        )

        # Başarılı aramadan sonra ayrı sonuç sayfasına
        # yönlendirme yapılmasını bekler.
        self.assertRedirects(
            response,
            reverse("search_results"),
            fetch_redirect_response=False,
        )

        # View tarafından oluşturulan güncel session verisini okur.
        search_state = self.client.session[
            "hugselect_search_results"
        ]

        # Sonuç sayfasının ihtiyaç duyacağı temel bilgilerin
        # session'a kaydedildiğini doğrular.
        self.assertEqual(
            search_state["query"],
            "English text-generation model",
        )
        self.assertEqual(
            search_state["results"],
            mock_results,
        )
        self.assertEqual(
            search_state["search_mode"],
            "feature-based",
        )
        self.assertEqual(
            search_state["moscow_requirements"],
            mock_results[0]["moscow_requirements"],
        )
    def test_displays_results_stored_in_session(self):
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "English text-generation model",
            "results": [
                {
                    "model_id": "author/model-a",
                    "basemodels": ["base/shared"],
                    "score": 90.0,
                },
                {
                    "model_id": "author/model-b",
                    "basemodels": ["base/shared"],
                    "score": 80.0,
                },
            ],
            "warning": None,
            "error": None,
            "search_mode": "feature-based",
        }

        session.save()

        # Ayrı sonuç sayfasını açar.
        response = self.client.get(
            reverse("search_results")
        )

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.context["query"],
            "English text-generation model",
        )

        self.assertEqual(
            response.context["search_mode"],
            "feature-based",
        )

        grouped_results = response.context["grouped_results"]

        self.assertEqual(
            grouped_results["single_base_model_groups"][0][
                "base_model"
            ],
            "base/shared",
        )

        self.assertEqual(
            len(
                grouped_results["single_base_model_groups"][0][
                    "results"
                ]
            ),
            2,
        )
    def test_preserves_family_filter_in_results_context(self):
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "English code-generation model",
            "results": [],
            "warning": None,
            "error": None,
            "search_mode": "feature-based",
            "search_scope": "family",
            "base_model_family": "llama",
        }
        session.save()

        response = self.client.get(
            reverse("search_results")
        )

        self.assertEqual(
            response.context["search_scope"],
            "family",
        )
        self.assertEqual(
            response.context["base_model_family"],
            "llama",
        )
        self.assertContains(
            response,
            "Base-model family: Llama",
        )

    def test_displays_structured_moscow_priority_summary(self):
        requirements = [
            {
                "feature_key": "language",
                "value": "English",
                "priority": "must",
                "moscow_priority": "Must Have",
                "constraint_kind": "hard_positive",
            },
            {
                "feature_key": "library_name",
                "value": "transformers",
                "priority": "strong_prefer",
                "moscow_priority": "Should Have",
                "constraint_kind": "soft_positive",
            },
            {
                "feature_key": "datasets",
                "value": "squad",
                "priority": "prefer",
                "moscow_priority": "Could Have",
                "constraint_kind": "soft_positive",
            },
            {
                "feature_key": "license_name",
                "value": "gpl-3.0",
                "priority": "avoid",
                "moscow_priority": "Won't Have",
                "constraint_kind": "hard_negative",
            },
        ]
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "English model",
            "results": [],
            "warning": None,
            "error": None,
            "search_mode": "feature-based",
            "search_scope": "all",
            "base_model_family": None,
            "moscow_requirements": requirements,
        }
        session.save()

        response = self.client.get(reverse("search_results"))

        self.assertEqual(
            response.context["moscow_requirements"],
            requirements,
        )
        for text in (
            "Must Have",
            "Should Have",
            "Could Have",
            "Won&#x27;t Have",
            "English",
            "transformers",
            "squad",
            "gpl-3.0",
        ):
            with self.subTest(text=text):
                self.assertContains(response, text, html=False)

    def test_basic_fallback_does_not_claim_moscow_reasoning(self):
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "English model",
            "results": [],
            "warning": None,
            "error": None,
            "search_mode": "basic-fallback",
            "search_scope": "all",
            "base_model_family": None,
            "moscow_requirements": [],
        }
        session.save()

        response = self.client.get(reverse("search_results"))

        self.assertNotContains(response, "Requirements / MoSCoW")
        self.assertContains(response, "Basic fallback search")

    def test_displays_all_models_scope(self):
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "English code-generation model",
            "results": [],
            "warning": None,
            "error": None,
            "search_mode": "feature-based",
            "search_scope": "all",
            "base_model_family": None,
        }
        session.save()

        response = self.client.get(
            reverse("search_results")
        )

        self.assertEqual(response.context["search_scope"], "all")
        self.assertIsNone(
            response.context["base_model_family"]
        )
        self.assertContains(response, "All models")


class AnalysisReportTests(TestCase):
    def _search_state(self):
        return {
            "query": "English text-generation model",
            "search_mode": "feature-based",
            "search_scope": "family",
            "base_model_family": "llama",
            "explicit_requirements": [
                {
                    "feature_key": "language",
                    "value": "English",
                    "priority": "must",
                },
                {
                    "feature_key": "license_name",
                    "value": "apache-2.0",
                    "priority": "should",
                },
            ],
            "results": [
                {
                    "model_id": "author/model-a",
                    "author": "author",
                    "pipeline_tag": "text-generation",
                    "license": "apache-2.0",
                    "library_name": "transformers",
                    "basemodels": ["meta-llama/Llama-3"],
                    "score": 91.25,
                    "availability": {
                        "status": "available",
                    },
                    "url": "https://huggingface.co/author/model-a",
                }
            ],
        }

    @patch("recommender.views.search_models_basic")
    @patch("recommender.views.search_models_feature_based")
    @patch("recommender.views.generate_analysis_report")
    def test_endpoint_returns_attachment_from_current_session_without_search(
        self,
        mock_generate_report,
        mock_feature_search,
        mock_basic_search,
    ):
        search_state = self._search_state()
        comparison_state = {
            "model_ids": ["author/model-a", "author/model-b"],
            "models": [{"model_id": "author/model-a"}],
            "coverage_rows": [],
        }
        stress_state = {
            "stress_summary": {
                "stability_label": "Stable winner",
            },
            "stress_result": {"scenarios": []},
        }
        session = self.client.session
        session["hugselect_search_results"] = search_state
        session["hugselect_comparison"] = comparison_state
        session["hugselect_decision_stress"] = stress_state
        session.save()
        mock_generate_report.return_value = b"%PDF-1.4\nmock report"

        response = self.client.get(reverse("analysis_report"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertEqual(
            response["Content-Disposition"],
            'attachment; filename="hugselect-analysis-report.pdf"',
        )
        report_kwargs = mock_generate_report.call_args.kwargs
        self.assertEqual(report_kwargs["search_state"], search_state)
        self.assertEqual(
            report_kwargs["search_state"]["query"],
            "English text-generation model",
        )
        self.assertEqual(
            report_kwargs["search_state"]["base_model_family"],
            "llama",
        )
        self.assertEqual(
            report_kwargs["search_state"]["explicit_requirements"],
            search_state["explicit_requirements"],
        )
        self.assertEqual(
            report_kwargs["search_state"]["results"],
            search_state["results"],
        )
        self.assertEqual(
            report_kwargs["comparison_state"],
            comparison_state,
        )
        self.assertEqual(
            report_kwargs["decision_stress_state"],
            stress_state,
        )
        mock_feature_search.assert_not_called()
        mock_basic_search.assert_not_called()

    def test_normalizes_all_available_report_sections(self):
        comparison_state = {
            "models": [
                {
                    "model_id": "author/model-a",
                    "comparison_score": 91.25,
                }
            ],
            "coverage_rows": [
                {
                    "feature_key": "language",
                    "user_value": "English",
                    "model_statuses": [
                        {
                            "model_id": "author/model-a",
                            "matched": True,
                        }
                    ],
                }
            ],
            "decision_summary": {
                "leaders": ["author/model-a"],
            },
        }
        stress_state = {
            "stress_summary": {
                "stability_label": "Stable winner",
                "stability_percentage": 100.0,
                "leaders": ["author/model-a"],
                "scenario_count": 5,
            },
            "stress_result": {
                "scenarios": [
                    {
                        "label": "Starting priorities",
                        "winners": ["author/model-a"],
                        "model_results": [],
                    }
                ]
            },
        }

        report_data = build_analysis_report_data(
            search_state=self._search_state(),
            comparison_state=comparison_state,
            decision_stress_state=stress_state,
        )

        self.assertEqual(
            report_data["overview"]["query"],
            "English text-generation model",
        )
        self.assertEqual(
            report_data["overview"]["base_model_family"],
            "llama",
        )
        self.assertEqual(
            report_data["explicit_requirements"]["must"][0],
            {"feature": "Language", "value": "English"},
        )
        self.assertEqual(
            report_data["explicit_requirements"]["should"][0],
            {"feature": "License Name", "value": "apache-2.0"},
        )
        self.assertEqual(
            report_data["recommended_models"][0]["model_id"],
            "author/model-a",
        )
        self.assertEqual(report_data["comparison"], comparison_state)
        self.assertEqual(report_data["decision_stress"], stress_state)

    def test_missing_optional_sections_generate_a_valid_pdf(self):
        pdf_bytes = generate_analysis_report(
            search_state={
                "query": "A summarization model",
                "search_scope": "all",
            },
            comparison_state=None,
            decision_stress_state=None,
        )

        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 1000)

    def test_results_page_exposes_download_action(self):
        session = self.client.session
        session["hugselect_search_results"] = self._search_state()
        session.save()

        response = self.client.get(reverse("search_results"))

        self.assertContains(response, "Download report")
        self.assertContains(
            response,
            f'href="{reverse("analysis_report")}"',
        )


class SearchFormFamilyFilterTests(TestCase):
    def test_displays_supported_family_options(self):
        response = self.client.get(reverse("search"))

        self.assertEqual(response.status_code, 200)

        for family, label in (
            ("llama", "Llama"),
            ("mistral", "Mistral"),
            ("qwen", "Qwen"),
            ("gemma", "Gemma"),
        ):
            with self.subTest(family=family):
                self.assertContains(
                    response,
                    f'<option value="{family}">{label}</option>',
                )

    def test_defaults_to_all_models_with_family_select_disabled(self):
        response = self.client.get(reverse("search"))
        response_html = response.content.decode()

        self.assertRegex(
            response_html,
            (
                r'<input\s+type="radio"\s+'
                r'name="search_scope"\s+'
                r'value="all"\s+checked'
            ),
        )
        self.assertRegex(
            response_html,
            (
                r'<select\s+id="base-model-family"\s+'
                r'name="base_model_family"\s+disabled'
            ),
        )


class ExplicitRequirementFormTests(TestCase):
    def test_form_exposes_supported_features_and_priorities(self):
        response = self.client.get(reverse("search"))
        response_html = response.content.decode()

        for _, feature_options in EXPLICIT_REQUIREMENT_FEATURE_GROUPS:
            for feature_key, feature_label in feature_options:
                with self.subTest(feature=feature_key):
                    self.assertRegex(
                        response_html,
                        (
                            rf'<option value="{feature_key}">\s+'
                            rf'{feature_label}\s+</option>'
                        ),
                    )

        for priority, priority_label in EXPLICIT_REQUIREMENT_PRIORITIES:
            with self.subTest(priority=priority):
                self.assertRegex(
                    response_html,
                    (
                        rf'<option value="{priority}">\s+'
                        rf'{priority_label.replace("'", "&#x27;")}'
                    ),
                )

        self.assertContains(response, "+ Add requirement")
        self.assertContains(response, 'class="remove-requirement"')
        self.assertContains(response, 'name="requirement_value"')

    def test_does_not_expose_downloads_or_likes_as_requirements(self):
        response = self.client.get(reverse("search"))

        self.assertNotContains(
            response,
            'value="downloads_last_30_days"',
        )
        self.assertNotContains(response, 'value="likes"')


class ExplicitRequirementParsingTests(SimpleTestCase):
    def test_parses_all_explicit_moscow_priorities(self):
        requirements = parse_explicit_requirements(
            ["task", "language", "Reliability", "gated"],
            ["text-generation", "English", "0.8", "true"],
            ["must", "should", "could", "wont"],
        )

        self.assertEqual(
            requirements,
            [
                {
                    "feature_key": "task",
                    "value": "text-generation",
                    "priority": "must",
                },
                {
                    "feature_key": "language",
                    "value": "English",
                    "priority": "should",
                },
                {
                    "feature_key": "Reliability",
                    "value": "0.8",
                    "priority": "could",
                },
                {
                    "feature_key": "gated",
                    "value": "true",
                    "priority": "wont",
                },
            ],
        )

    def test_ignores_an_unselected_empty_row(self):
        self.assertEqual(
            parse_explicit_requirements(
                [""],
                [""],
                ["must"],
            ),
            [],
        )

    def test_rejects_duplicate_requirement(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Duplicate requirement: Language = english\.",
        ):
            parse_explicit_requirements(
                ["language", "language"],
                ["English", "english"],
                ["must", "must"],
            )

    def test_rejects_unsupported_feature_priority_and_missing_value(self):
        invalid_cases = (
            (
                ["downloads_last_30_days"],
                ["1000"],
                ["must"],
                "unsupported feature",
            ),
            (
                ["language"],
                ["English"],
                ["urgent"],
                "unsupported priority",
            ),
            (
                ["language"],
                [""],
                ["must"],
                "must include a value",
            ),
        )

        for features, values, priorities, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parse_explicit_requirements(
                        features,
                        values,
                        priorities,
                    )

    def test_rejects_contradictory_requirement(self):
        with self.assertRaisesRegex(
            ValueError,
            (
                r"Contradictory requirements: Language = English "
                r"cannot be both MUST and WON'T\."
            ),
        ):
            parse_explicit_requirements(
                ["language", "language"],
                ["English", "English"],
                ["must", "wont"],
            )


class MoscowPrioritySemanticsTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.builder = _make_feature_search_builder(
            limit=AVAILABILITY_CANDIDATE_LIMIT
        )

    def _group(
        self,
        priority,
        value,
        field="Metadata.language",
        *,
        base_weight=1.0,
        explicit_priority=None,
    ):
        return FeatureGroup(
            feature_key="language",
            priority=priority,
            include=[value],
            fields=[field],
            base_weight=base_weight,
            explicit_priority=explicit_priority,
        )

    def test_maps_existing_priorities_to_moscow(self):
        self.assertEqual(
            PRIORITY_TO_MOSCOW,
            {
                "must": {
                    "label": "Must Have",
                    "kind": "hard_positive",
                },
                "strong_prefer": {
                    "label": "Should Have",
                    "kind": "soft_positive",
                },
                "prefer": {
                    "label": "Could Have",
                    "kind": "soft_positive",
                },
                "avoid": {
                    "label": "Won't Have",
                    "kind": "hard_negative",
                },
            },
        )

    def test_must_match_remains_eligible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"language": ["English"]}},
            prebuilt_groups=[self._group("must", "English")],
        )

        self.assertTrue(explanation["hard_filters_passed"])
        self.assertEqual(
            explanation["hard_filters"][0]["moscow_priority"],
            "Must Have",
        )

    def test_must_non_match_is_infeasible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"language": ["French"]}},
            prebuilt_groups=[self._group("must", "English")],
        )

        hard_result = explanation["hard_filters"][0]
        self.assertFalse(explanation["hard_filters_passed"])
        self.assertEqual(hard_result["evidence_status"], "non_match")
        self.assertEqual(
            hard_result["exclusion_reason"],
            "must_not_satisfied",
        )

    def test_missing_must_evidence_is_not_called_a_confirmed_non_match(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {}},
            prebuilt_groups=[self._group("must", "English")],
        )

        hard_result = explanation["hard_filters"][0]
        self.assertFalse(explanation["hard_filters_passed"])
        self.assertEqual(
            hard_result["evidence_status"],
            "missing_evidence",
        )
        self.assertEqual(
            hard_result["exclusion_reason"],
            "missing_evidence",
        )

    def test_wont_match_is_infeasible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"license": "gpl-3.0"}},
            prebuilt_groups=[
                self._group("avoid", "gpl-3.0", "Metadata.license")
            ],
        )

        hard_result = explanation["hard_filters"][0]
        self.assertFalse(explanation["hard_filters_passed"])
        self.assertEqual(
            hard_result["exclusion_reason"],
            "prohibited_value_matched",
        )

    def test_wont_non_match_remains_eligible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"license": "apache-2.0"}},
            prebuilt_groups=[
                self._group("avoid", "gpl-3.0", "Metadata.license")
            ],
        )

        hard_result = explanation["hard_filters"][0]
        self.assertTrue(explanation["hard_filters_passed"])
        self.assertEqual(hard_result["evidence_status"], "non_match")
        self.assertIsNone(hard_result["exclusion_reason"])

    def test_missing_wont_evidence_does_not_exclude_or_claim_non_match(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {}},
            prebuilt_groups=[
                self._group("avoid", "gpl-3.0", "Metadata.license")
            ],
        )

        hard_result = explanation["hard_filters"][0]
        self.assertTrue(explanation["hard_filters_passed"])
        self.assertEqual(
            hard_result["evidence_status"],
            "missing_evidence",
        )
        self.assertIsNone(hard_result["exclusion_reason"])

    def test_missing_should_and_could_do_not_exclude(self):
        for priority, label in (
            ("strong_prefer", "Should Have"),
            ("prefer", "Could Have"),
        ):
            with self.subTest(priority=priority):
                explanation = self.builder.compare_bundle_to_sample(
                    None,
                    {"Metadata": {}},
                    prebuilt_groups=[
                        self._group(priority, "English")
                    ],
                )
                self.assertTrue(explanation["hard_filters_passed"])
                self.assertEqual(
                    explanation["per_feature"][0]["moscow_priority"],
                    label,
                )
                self.assertFalse(
                    explanation["per_feature"][0]["matches"][0][
                        "matched"
                    ]
                )

    def test_should_match_outweighs_comparable_could_match(self):
        should_group = self._group(
            "strong_prefer",
            "English",
            base_weight=9.5,
            explicit_priority="should",
        )
        could_group = self._group(
            "prefer",
            "apache-2.0",
            "Metadata.license",
            base_weight=9.5,
            explicit_priority="could",
        )

        should_model = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"language": ["English"]}},
            prebuilt_groups=[should_group, could_group],
        )
        could_model = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"license": "apache-2.0"}},
            prebuilt_groups=[should_group, could_group],
        )

        self.assertEqual(
            should_model["per_feature"][0]["effective_weight"],
            9.5,
        )
        self.assertEqual(
            could_model["per_feature"][1]["effective_weight"],
            4.75,
        )
        self.assertGreater(
            should_model["total_match_score"],
            could_model["total_match_score"],
        )

    def test_explicit_soft_multiplier_configuration_is_exact(self):
        self.assertEqual(
            self.builder.feature_weight_groups["preference"]["language"],
            9.5,
        )
        self.assertEqual(
            EXPLICIT_MOSCOW_PRIORITIES["should"]["multiplier"],
            1.0,
        )
        self.assertEqual(
            EXPLICIT_MOSCOW_PRIORITIES["could"]["multiplier"],
            0.5,
        )

    def test_explicit_priority_overrides_extraction_without_double_counting(self):
        extracted_group = self._group("prefer", "English")
        merged_groups = self.builder.apply_explicit_requirements(
            [extracted_group],
            [
                {
                    "feature_key": "language",
                    "value": "English",
                    "priority": "must",
                }
            ],
        )

        self.assertEqual(len(merged_groups), 1)
        self.assertEqual(merged_groups[0].include, ["English"])
        self.assertEqual(merged_groups[0].explicit_priority, "must")
        self.assertEqual(merged_groups[0].priority, "must")
        self.assertEqual(
            len(
                self.builder.summarize_moscow_requirements(
                    merged_groups
                )
            ),
            1,
        )

    def test_no_explicit_requirements_preserves_extracted_groups(self):
        extracted_groups = [self._group("prefer", "English")]

        merged_groups = self.builder.apply_explicit_requirements(
            extracted_groups,
            [],
        )

        self.assertIs(merged_groups, extracted_groups)

    def test_explicit_must_missing_is_infeasible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {}},
            prebuilt_groups=[
                self._group(
                    "must",
                    "English",
                    explicit_priority="must",
                )
            ],
        )

        self.assertFalse(explanation["hard_filters_passed"])
        self.assertEqual(
            explanation["hard_filters"][0]["priority"],
            "must",
        )
        self.assertEqual(
            explanation["hard_filters"][0]["priority_source"],
            "explicit",
        )

    def test_explicit_wont_match_is_infeasible(self):
        explanation = self.builder.compare_bundle_to_sample(
            None,
            {"Metadata": {"license": "gpl-3.0"}},
            prebuilt_groups=[
                self._group(
                    "avoid",
                    "gpl-3.0",
                    "Metadata.license",
                    explicit_priority="wont",
                )
            ],
        )

        self.assertFalse(explanation["hard_filters_passed"])
        self.assertEqual(
            explanation["hard_filters"][0]["exclusion_reason"],
            "prohibited_value_matched",
        )

    def test_explicit_should_and_could_non_matches_remain_feasible(self):
        for explicit_priority, legacy_priority in (
            ("should", "strong_prefer"),
            ("could", "prefer"),
        ):
            with self.subTest(priority=explicit_priority):
                explanation = self.builder.compare_bundle_to_sample(
                    None,
                    {"Metadata": {}},
                    prebuilt_groups=[
                        self._group(
                            legacy_priority,
                            "English",
                            explicit_priority=explicit_priority,
                        )
                    ],
                )

                self.assertTrue(explanation["hard_filters_passed"])
                self.assertFalse(
                    explanation["per_feature"][0]["matches"][0][
                        "matched"
                    ]
                )

    def test_hard_constraints_are_inside_query_before_candidate_limit(self):
        must_group = self._group("must", "English")
        wont_group = self._group(
            "avoid",
            "gpl-3.0",
            "Metadata.license",
        )
        should_group = self._group("strong_prefer", "German")
        family_filter = _base_model_family_filter("gemma")

        query = self.builder.build_query(
            [must_group, wont_group, should_group],
            extra_filter_clauses=[family_filter],
        )
        ranked_query = query["query"]
        if "function_score" in ranked_query:
            ranked_query = ranked_query["function_score"]["query"]
        bool_query = ranked_query["bool"]

        self.assertEqual(
            query["size"],
            AVAILABILITY_CANDIDATE_LIMIT,
        )
        self.assertIn(family_filter, bool_query["filter"])
        self.assertEqual(len(bool_query["filter"]), 2)
        self.assertEqual(len(bool_query["must_not"]), 1)
        self.assertEqual(bool_query["minimum_should_match"], 0)

    def test_explicit_hard_constraints_use_the_same_early_query_filters(self):
        query = self.builder.build_query(
            [
                self._group(
                    "must",
                    "English",
                    explicit_priority="must",
                ),
                self._group(
                    "avoid",
                    "gpl-3.0",
                    "Metadata.license",
                    explicit_priority="wont",
                ),
            ]
        )
        ranked_query = query["query"]
        if "function_score" in ranked_query:
            ranked_query = ranked_query["function_score"]["query"]

        self.assertEqual(
            query["size"],
            AVAILABILITY_CANDIDATE_LIMIT,
        )
        self.assertEqual(len(ranked_query["bool"]["filter"]), 1)
        self.assertEqual(
            len(ranked_query["bool"]["must_not"]),
            1,
        )


class MoscowFeatureSearchTests(SimpleTestCase):
    @patch("recommender.services._make_feature_search_builder")
    @patch("recommender.services._extract_feature_bundle")
    @patch("recommender.services.Elasticsearch")
    def test_only_feasible_hits_continue_to_results(
        self,
        mock_elasticsearch,
        mock_extract_feature_bundle,
        mock_make_builder,
    ):
        mock_es = mock_elasticsearch.return_value
        mock_es.indices.exists.return_value = True
        mock_extract_feature_bundle.return_value = object()

        infeasible_hit = {
            "_id": "author__blocked.json",
            "_source": {"modelID": "author/blocked"},
        }
        feasible_hit = {
            "_id": "author__eligible.json",
            "_source": {"modelID": "author/eligible"},
        }
        mock_builder = mock_make_builder.return_value
        mock_builder.precompute_feature_group_cache.return_value = []
        mock_builder.search.return_value = (
            {"hits": {"hits": [infeasible_hit, feasible_hit]}},
            {},
            [],
        )
        mock_builder.summarize_moscow_requirements.return_value = [
            {
                "feature_key": "language",
                "value": "English",
                "priority": "must",
                "moscow_priority": "Must Have",
                "constraint_kind": "hard_positive",
            }
        ]
        mock_builder.compare_bundle_to_sample.side_effect = [
            {"hard_filters_passed": False},
            {"hard_filters_passed": True},
        ]

        results = search_models_feature_based("English model", limit=10)

        self.assertEqual(
            [result["model_id"] for result in results],
            ["author/eligible"],
        )
        self.assertEqual(
            results[0]["moscow_requirements"][0]["moscow_priority"],
            "Must Have",
        )

    @patch("recommender.services._make_feature_search_builder")
    @patch("recommender.services._extract_feature_bundle")
    @patch("recommender.services.Elasticsearch")
    def test_explicit_requirements_are_merged_before_search(
        self,
        mock_elasticsearch,
        mock_extract_feature_bundle,
        mock_make_builder,
    ):
        mock_es = mock_elasticsearch.return_value
        mock_es.indices.exists.return_value = True
        bundle = mock_extract_feature_bundle.return_value
        extracted_groups = [object()]
        merged_groups = [object()]
        explicit_requirements = [
            {
                "feature_key": "language",
                "value": "English",
                "priority": "must",
            }
        ]

        mock_builder = mock_make_builder.return_value
        mock_builder.precompute_feature_group_cache.return_value = (
            extracted_groups
        )
        mock_builder.apply_explicit_requirements.return_value = (
            merged_groups
        )
        mock_builder.search.return_value = (
            {"hits": {"hits": []}},
            {},
            merged_groups,
        )
        mock_builder.summarize_moscow_requirements.return_value = []

        search_models_feature_based(
            "English model",
            explicit_requirements=explicit_requirements,
        )

        mock_builder.apply_explicit_requirements.assert_called_once_with(
            extracted_groups,
            explicit_requirements,
        )
        mock_builder.search.assert_called_once_with(
            es_client=mock_es,
            index="models_t7",
            features=bundle,
            prebuilt_groups=merged_groups,
            include_explain=False,
            extra_filter_clauses=None,
        )


class BaseModelFamilyFilterTests(SimpleTestCase):
    def test_builds_supported_family_filters(self):
        for family, pattern in (
            ("llama", "*llama*"),
            ("mistral", "*mistral*"),
            ("qwen", "*qwen*"),
            ("gemma", "*gemma*"),
        ):
            with self.subTest(family=family):
                self.assertEqual(
                    _base_model_family_filter(family),
                    {
                        "wildcard": {
                            "Metadata.basemodels": {
                                "value": pattern,
                                "case_insensitive": True,
                            }
                        }
                    }
                )

    def test_returns_none_without_family(self):
        self.assertIsNone(
            _base_model_family_filter(None)
        )

    def test_returns_none_for_unsupported_family(self):
        self.assertIsNone(
            _base_model_family_filter("arbitrary-family")
        )


class BasicSearchFamilyFilterTests(SimpleTestCase):
    @patch("recommender.services.Elasticsearch")
    def test_applies_family_filter_before_result_limit(
        self,
        mock_elasticsearch,
    ):
        mock_es = mock_elasticsearch.return_value
        mock_es.search.return_value = {
            "hits": {"hits": []}
        }

        search_models_basic(
            "English text-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family="qwen",
        )

        search_body = mock_es.search.call_args.kwargs["body"]
        self.assertEqual(
            search_body["size"],
            AVAILABILITY_CANDIDATE_LIMIT,
        )
        self.assertIn(
            "Metadata.basemodels",
            search_body["_source"],
        )
        self.assertEqual(
            search_body["query"]["bool"]["filter"],
            [
                {
                    "wildcard": {
                        "Metadata.basemodels": {
                            "value": "*qwen*",
                            "case_insensitive": True,
                        }
                    }
                }
            ],
        )
        self.assertEqual(
            search_body["query"]["bool"]["must"][0][
                "multi_match"
            ]["query"],
            "English text-generation model",
        )

    @patch("recommender.services.Elasticsearch")
    def test_preserves_basic_text_query_without_family(
        self,
        mock_elasticsearch,
    ):
        mock_es = mock_elasticsearch.return_value
        mock_es.search.return_value = {
            "hits": {"hits": []}
        }

        search_models_basic(
            "English text-generation model",
            limit=10,
        )

        search_body = mock_es.search.call_args.kwargs["body"]
        self.assertIn("multi_match", search_body["query"])
        self.assertNotIn("bool", search_body["query"])


class FeatureQueryBuilderFilterTests(SimpleTestCase):
    def test_places_extra_filter_inside_query_before_limit(self):
        builder = _make_feature_search_builder(
            limit=AVAILABILITY_CANDIDATE_LIMIT
        )
        family_filter = _base_model_family_filter("gemma")

        query = builder.build_query(
            [],
            extra_filter_clauses=[family_filter],
        )
        filtered_query = query["query"]

        if "function_score" in filtered_query:
            filtered_query = filtered_query[
                "function_score"
            ]["query"]

        self.assertEqual(
            query["size"],
            AVAILABILITY_CANDIDATE_LIMIT,
        )
        self.assertEqual(
            filtered_query["bool"]["filter"],
            [family_filter],
        )


class FeatureSearchFamilyFilterTests(SimpleTestCase):
    @patch("recommender.services._make_feature_search_builder")
    @patch("recommender.services._extract_feature_bundle")
    @patch("recommender.services.Elasticsearch")
    def test_passes_llama_filter_to_builder_search(
        self,
        mock_elasticsearch,
        mock_extract_feature_bundle,
        mock_make_builder,
    ):
        mock_es = mock_elasticsearch.return_value
        mock_es.indices.exists.return_value = True

        mock_extract_feature_bundle.return_value = object()

        mock_builder = mock_make_builder.return_value
        mock_builder.precompute_feature_group_cache.return_value = []
        mock_builder.search.return_value = (
            {"hits": {"hits": []}},
            {},
            [],
        )

        search_models_feature_based(
            "English text-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family="llama",
        )

        mock_make_builder.assert_called_once_with(
            limit=AVAILABILITY_CANDIDATE_LIMIT
        )

        mock_builder.search.assert_called_once_with(
            es_client=mock_es,
            index="models_t7",
            features=mock_extract_feature_bundle.return_value,
            prebuilt_groups=[],
            include_explain=False,
            extra_filter_clauses=[
                {
                    "wildcard": {
                        "Metadata.basemodels": {
                            "value": "*llama*",
                            "case_insensitive": True,
                        }
                    }
                }
            ],
        )
class SearchViewFamilyFilterTests(TestCase):
    @patch("recommender.views.search_models_feature_based")
    def test_passes_selected_family_to_feature_search(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = []

        for submitted_family, expected_family in (
            ("Llama", "llama"),
            ("Mistral", "mistral"),
            ("QWEN", "qwen"),
            ("gemma", "gemma"),
        ):
            with self.subTest(family=expected_family):
                mock_search_models_feature_based.reset_mock()

                self.client.post(
                    reverse("search"),
                    {
                        "query": "English code-generation model",
                        "search_scope": "family",
                        "base_model_family": submitted_family,
                    },
                )

                mock_search_models_feature_based.assert_called_once_with(
                    "English code-generation model",
                    limit=AVAILABILITY_CANDIDATE_LIMIT,
                    base_model_family=expected_family,
                )

    @patch("recommender.views.search_models_feature_based")
    def test_all_models_passes_no_family_filter(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = []

        self.client.post(
            reverse("search"),
            {
                "query": "English code-generation model",
                "search_scope": "all",
                "base_model_family": "qwen",
            },
        )

        mock_search_models_feature_based.assert_called_once_with(
            "English code-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family=None,
        )
        search_state = self.client.session[
            "hugselect_search_results"
        ]
        self.assertEqual(search_state["search_scope"], "all")
        self.assertIsNone(search_state["base_model_family"])

    @patch("recommender.views.search_models_feature_based")
    def test_rejects_missing_or_unsupported_family(
        self,
        mock_search_models_feature_based,
    ):
        for submitted_family in ("", "unsupported-family"):
            with self.subTest(family=submitted_family):
                self.client.post(
                    reverse("search"),
                    {
                        "query": "English code-generation model",
                        "search_scope": "family",
                        "base_model_family": submitted_family,
                    },
                )

                search_state = self.client.session[
                    "hugselect_search_results"
                ]
                self.assertEqual(
                    search_state["error"],
                    "Please select a supported base-model family.",
                )
                self.assertIsNone(
                    search_state["base_model_family"]
                )

        mock_search_models_feature_based.assert_not_called()

    @patch("recommender.views.search_models_basic")
    @patch("recommender.views.search_models_feature_based")
    def test_passes_family_to_basic_fallback(
        self,
        mock_search_models_feature_based,
        mock_search_models_basic,
    ):
        mock_search_models_feature_based.side_effect = RuntimeError(
            "Feature search unavailable"
        )
        mock_search_models_basic.return_value = []

        self.client.post(
            reverse("search"),
            {
                "query": "English code-generation model",
                "search_scope": "family",
                "base_model_family": "Mistral",
            },
        )

        mock_search_models_basic.assert_called_once_with(
            "English code-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family="mistral",
        )


class ExplicitRequirementViewTests(TestCase):
    @patch("recommender.views.search_models_feature_based")
    def test_passes_parsed_explicit_requirements_to_feature_search(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = []
        explicit_requirements = [
            {
                "feature_key": "task",
                "value": "text-generation",
                "priority": "must",
            },
            {
                "feature_key": "language",
                "value": "English",
                "priority": "should",
            },
            {
                "feature_key": "Reliability",
                "value": "0.8",
                "priority": "could",
            },
            {
                "feature_key": "gated",
                "value": "true",
                "priority": "wont",
            },
        ]

        response = self.client.post(
            reverse("search"),
            {
                "query": "English text-generation model",
                "requirement_feature": [
                    requirement["feature_key"]
                    for requirement in explicit_requirements
                ],
                "requirement_value": [
                    requirement["value"]
                    for requirement in explicit_requirements
                ],
                "requirement_priority": [
                    requirement["priority"]
                    for requirement in explicit_requirements
                ],
            },
        )

        self.assertRedirects(
            response,
            reverse("search_results"),
            fetch_redirect_response=False,
        )
        mock_search_models_feature_based.assert_called_once_with(
            "English text-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family=None,
            explicit_requirements=explicit_requirements,
        )
        self.assertEqual(
            self.client.session["hugselect_search_results"][
                "explicit_requirements"
            ],
            explicit_requirements,
        )

    @patch("recommender.views.search_models_feature_based")
    def test_contradictory_requirements_return_clear_validation_error(
        self,
        mock_search_models_feature_based,
    ):
        self.client.post(
            reverse("search"),
            {
                "query": "English model",
                "requirement_feature": ["language", "language"],
                "requirement_value": ["English", "English"],
                "requirement_priority": ["must", "wont"],
            },
        )

        search_state = self.client.session[
            "hugselect_search_results"
        ]
        self.assertEqual(
            search_state["error"],
            (
                "Contradictory requirements: Language = English "
                "cannot be both MUST and WON'T."
            ),
        )
        self.assertEqual(search_state["explicit_requirements"], [])
        mock_search_models_feature_based.assert_not_called()

    @patch("recommender.views.search_models_feature_based")
    def test_no_explicit_rows_preserves_natural_language_call(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = []

        self.client.post(
            reverse("search"),
            {
                "query": "English model",
                "requirement_feature": [""],
                "requirement_value": [""],
                "requirement_priority": ["must"],
            },
        )

        mock_search_models_feature_based.assert_called_once_with(
            "English model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family=None,
        )

    @patch("recommender.views.search_models_feature_based")
    def test_results_page_displays_explicit_priorities_from_session(
        self,
        mock_search_models_feature_based,
    ):
        mock_search_models_feature_based.return_value = []

        response = self.client.post(
            reverse("search"),
            {
                "query": "English model",
                "requirement_feature": [
                    "task",
                    "language",
                    "Reliability",
                    "gated",
                ],
                "requirement_value": [
                    "text-generation",
                    "English",
                    "0.8",
                    "true",
                ],
                "requirement_priority": [
                    "must",
                    "should",
                    "could",
                    "wont",
                ],
            },
            follow=True,
        )

        self.assertTemplateUsed(
            response,
            "recommender/search_results.html",
        )
        self.assertContains(response, "Explicit priorities")
        for text in (
            "MUST",
            "SHOULD",
            "COULD",
            "WON&#x27;T",
            "Task:",
            "text-generation",
            "Language:",
            "English",
            "Reliability:",
            "0.8",
            "Gated:",
            "true",
        ):
            with self.subTest(text=text):
                self.assertContains(response, text)


class ModelAvailabilityCheckTests(SimpleTestCase):
    @patch("recommender.services.urlopen")
    def test_http_200_is_available(self, mock_urlopen):
        response = mock_urlopen.return_value.__enter__.return_value
        response.status = 200

        availability = check_model_availability(
            "https://huggingface.co/author/model"
        )

        self.assertEqual(
            availability,
            {"status": "available", "http_status": 200},
        )
        self.assertEqual(
            mock_urlopen.call_args.kwargs["timeout"],
            3.0,
        )

    @patch("recommender.services.urlopen")
    def test_redirect_ending_in_http_200_is_available(
        self,
        mock_urlopen,
    ):
        response = mock_urlopen.return_value.__enter__.return_value
        response.status = 200
        response.geturl.return_value = (
            "https://huggingface.co/new-author/new-model"
        )

        availability = check_model_availability(
            "https://huggingface.co/old-author/old-model"
        )

        self.assertEqual(availability["status"], "available")
        self.assertEqual(availability["http_status"], 200)

    @patch("recommender.services.urlopen")
    def test_http_404_and_410_are_unavailable(
        self,
        mock_urlopen,
    ):
        for http_status in (404, 410):
            with self.subTest(http_status=http_status):
                mock_urlopen.side_effect = HTTPError(
                    "https://huggingface.co/missing/model",
                    http_status,
                    "missing",
                    None,
                    None,
                )

                self.assertEqual(
                    check_model_availability(
                        "https://huggingface.co/missing/model"
                    ),
                    {
                        "status": "unavailable",
                        "http_status": http_status,
                    },
                )

    @patch("recommender.services.urlopen")
    def test_ambiguous_http_errors_are_unknown(
        self,
        mock_urlopen,
    ):
        for http_status in (401, 403, 429, 500):
            with self.subTest(http_status=http_status):
                mock_urlopen.side_effect = HTTPError(
                    "https://huggingface.co/author/model",
                    http_status,
                    "ambiguous",
                    None,
                    None,
                )

                self.assertEqual(
                    check_model_availability(
                        "https://huggingface.co/author/model"
                    ),
                    {
                        "status": "unknown",
                        "http_status": http_status,
                    },
                )

    @patch("recommender.services.urlopen")
    def test_timeout_and_network_failure_are_unknown(
        self,
        mock_urlopen,
    ):
        for failure in (
            TimeoutError("timed out"),
            URLError("network unavailable"),
        ):
            with self.subTest(failure=type(failure).__name__):
                mock_urlopen.side_effect = failure

                self.assertEqual(
                    check_model_availability(
                        "https://huggingface.co/author/model"
                    ),
                    {"status": "unknown", "http_status": None},
                )


class AvailableResultSelectionTests(SimpleTestCase):
    def test_skips_unavailable_results_without_reordering(self):
        ranked_results = [
            {"model_id": model_id, "score": score}
            for model_id, score in (
                ("A", 50),
                ("B", 40),
                ("C", 30),
                ("D", 20),
                ("E", 10),
            )
        ]
        statuses = {
            "A": "available",
            "B": "unavailable",
            "C": "available",
            "D": "unavailable",
            "E": "available",
        }

        def checker(url):
            model_id = url.rsplit("/", 1)[-1]
            status = statuses[model_id]
            return {
                "status": status,
                "http_status": 200 if status == "available" else 404,
            }

        selected, summary = select_available_results(
            ranked_results,
            availability_checker=checker,
        )

        self.assertEqual(
            [result["model_id"] for result in selected],
            ["A", "C", "E"],
        )
        self.assertEqual(
            [result["score"] for result in selected],
            [50, 30, 10],
        )
        self.assertEqual(summary["unavailable_count"], 2)
        self.assertEqual(summary["shortfall"], 7)

    def test_later_candidates_replace_unavailable_top_ten(self):
        ranked_results = [
            {"model_id": f"author/model-{rank}", "score": 101 - rank}
            for rank in range(1, 13)
        ]
        checker = Mock(
            side_effect=lambda url: {
                "status": (
                    "unavailable"
                    if url.endswith(("model-2", "model-5"))
                    else "available"
                ),
                "http_status": (
                    404
                    if url.endswith(("model-2", "model-5"))
                    else 200
                ),
            }
        )

        selected, summary = select_available_results(
            ranked_results,
            availability_checker=checker,
        )

        self.assertEqual(len(selected), 10)
        self.assertEqual(
            [result["model_id"] for result in selected],
            [
                "author/model-1",
                "author/model-3",
                "author/model-4",
                "author/model-6",
                "author/model-7",
                "author/model-8",
                "author/model-9",
                "author/model-10",
                "author/model-11",
                "author/model-12",
            ],
        )
        self.assertEqual(
            [result["score"] for result in selected],
            [100, 98, 97, 95, 94, 93, 92, 91, 90, 89],
        )
        self.assertEqual(summary["shortfall"], 0)
        self.assertEqual(checker.call_count, 12)

    def test_unknown_and_checker_failure_are_skipped_safely(self):
        ranked_results = [
            {"model_id": "author/unknown"},
            {"model_id": "author/failure"},
            {"model_id": "author/available"},
        ]

        def checker(url):
            if url.endswith("unknown"):
                return {"status": "unknown", "http_status": 429}
            if url.endswith("failure"):
                raise OSError("connection failed")
            return {"status": "available", "http_status": 200}

        selected, summary = select_available_results(
            ranked_results,
            availability_checker=checker,
        )

        self.assertEqual(
            [result["model_id"] for result in selected],
            ["author/available"],
        )
        self.assertEqual(summary["unknown_count"], 2)
        self.assertEqual(summary["unavailable_count"], 0)

    def test_stops_after_ten_available_results(self):
        checker = Mock(
            return_value={"status": "available", "http_status": 200}
        )
        ranked_results = [
            {"model_id": f"author/model-{rank}"}
            for rank in range(1, 21)
        ]

        selected, summary = select_available_results(
            ranked_results,
            availability_checker=checker,
        )

        self.assertEqual(len(selected), 10)
        self.assertEqual(checker.call_count, 10)
        self.assertEqual(summary["http_check_count"], 10)

    def test_checks_duplicate_urls_only_once(self):
        checker = Mock(
            return_value={"status": "available", "http_status": 200}
        )
        ranked_results = [
            {"model_id": "author/model"},
            {"model_id": "author/model"},
        ]

        selected, summary = select_available_results(
            ranked_results,
            availability_checker=checker,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(checker.call_count, 1)
        self.assertEqual(summary["http_check_count"], 1)


class SearchAvailabilityIntegrationTests(TestCase):
    @patch("recommender.services.check_model_availability")
    @patch("recommender.views.search_models_feature_based")
    def test_feature_search_backfills_verified_results_before_grouping(
        self,
        mock_search_models_feature_based,
        mock_check_model_availability,
    ):
        candidate_results = [
            {
                "model_id": f"author/model-{rank}",
                "url": f"https://huggingface.co/author/model-{rank}",
                "basemodels": ["meta-llama/shared"],
                "score": 101 - rank,
            }
            for rank in range(1, 13)
        ]
        mock_search_models_feature_based.return_value = candidate_results

        def availability(url):
            unavailable = url.endswith(("model-2", "model-5"))
            return {
                "status": "unavailable" if unavailable else "available",
                "http_status": 404 if unavailable else 200,
            }

        mock_check_model_availability.side_effect = availability

        response = self.client.post(
            reverse("search"),
            {
                "query": "English code-generation model",
                "search_scope": "family",
                "base_model_family": "llama",
            },
            follow=True,
        )

        mock_search_models_feature_based.assert_called_once_with(
            "English code-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family="llama",
        )
        search_state = self.client.session[
            "hugselect_search_results"
        ]
        self.assertEqual(len(search_state["results"]), 10)
        self.assertEqual(
            [result["model_id"] for result in search_state["results"]],
            [
                "author/model-1",
                "author/model-3",
                "author/model-4",
                "author/model-6",
                "author/model-7",
                "author/model-8",
                "author/model-9",
                "author/model-10",
                "author/model-11",
                "author/model-12",
            ],
        )
        self.assertTrue(
            all(
                result["availability"]["status"] == "available"
                for result in search_state["results"]
            )
        )
        self.assertEqual(
            len(
                response.context["grouped_results"]
                ["single_base_model_groups"][0]["results"]
            ),
            10,
        )
        self.assertNotContains(response, "author/model-2")
        self.assertContains(response, "author/model-12")

    @patch("recommender.services.check_model_availability")
    @patch("recommender.views.search_models_basic")
    @patch("recommender.views.search_models_feature_based")
    def test_basic_fallback_uses_larger_candidate_pool(
        self,
        mock_search_models_feature_based,
        mock_search_models_basic,
        mock_check_model_availability,
    ):
        mock_search_models_feature_based.side_effect = RuntimeError(
            "feature search unavailable"
        )
        mock_search_models_basic.return_value = [
            {
                "model_id": f"author/model-{rank}",
                "url": f"https://huggingface.co/author/model-{rank}",
                "score": 20 - rank,
            }
            for rank in range(1, 16)
        ]
        mock_check_model_availability.return_value = {
            "status": "available",
            "http_status": 200,
        }

        self.client.post(
            reverse("search"),
            {
                "query": "English text-generation model",
                "search_scope": "family",
                "base_model_family": "qwen",
            },
        )

        mock_search_models_basic.assert_called_once_with(
            "English text-generation model",
            limit=AVAILABILITY_CANDIDATE_LIMIT,
            base_model_family="qwen",
        )
        search_state = self.client.session[
            "hugselect_search_results"
        ]
        self.assertEqual(len(search_state["results"]), 10)
        self.assertEqual(
            search_state["warning"],
            (
                "Advanced recommendation is temporarily unavailable. "
                "Showing basic search results instead."
            ),
        )

    def test_results_page_shows_availability_and_shortfall_warning(self):
        session = self.client.session
        session["hugselect_search_results"] = {
            "query": "available model",
            "results": [
                {
                    "model_id": "author/model",
                    "availability": {
                        "status": "available",
                        "http_status": 200,
                    },
                    "score": 90.0,
                }
            ],
            "warning": (
                "Only 1 model could be verified as currently available."
            ),
            "error": None,
            "search_mode": "feature-based",
            "search_scope": "all",
            "base_model_family": None,
            "availability_summary": {
                "candidate_count": 3,
                "available_count": 1,
                "shortfall": 9,
            },
        }
        session.save()

        response = self.client.get(reverse("search_results"))

        self.assertContains(response, "Available")
        self.assertContains(
            response,
            "Only 1 model could be verified as currently available.",
        )
        self.assertContains(response, 'id="compare-form"')
        self.assertContains(
            response,
            'id="search-task-tooltip-result-1"',
        )

    @patch("recommender.services.check_model_availability")
    @patch("recommender.views.search_models_feature_based")
    def test_search_warns_when_candidate_pool_cannot_fill_ten(
        self,
        mock_search_models_feature_based,
        mock_check_model_availability,
    ):
        mock_search_models_feature_based.return_value = [
            {
                "model_id": f"author/model-{rank}",
                "url": f"https://huggingface.co/author/model-{rank}",
            }
            for rank in range(1, 6)
        ]
        mock_check_model_availability.side_effect = [
            {"status": "available", "http_status": 200},
            {"status": "unavailable", "http_status": 404},
            {"status": "unknown", "http_status": 429},
            {"status": "available", "http_status": 200},
            {"status": "available", "http_status": 200},
        ]

        self.client.post(
            reverse("search"),
            {"query": "English model"},
        )

        search_state = self.client.session[
            "hugselect_search_results"
        ]
        self.assertEqual(len(search_state["results"]), 3)
        self.assertEqual(
            search_state["warning"],
            "Only 3 models could be verified as currently available.",
        )
        self.assertEqual(
            search_state["availability_summary"]["unknown_count"],
            1,
        )
        self.assertEqual(
            search_state["availability_summary"]["unavailable_count"],
            1,
        )
