from django.test import SimpleTestCase
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from .services import build_model_graph
from .decision_stress import (
    SCENARIOS,
    category_for_feature,
    run_decision_stress_test,
    score_explanation_for_scenario,
    summarize_decision_stress_test,

)

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
            ["text generation"],
        )

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