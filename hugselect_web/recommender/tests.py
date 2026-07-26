from django.test import SimpleTestCase
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from .services import build_model_graph
from .decision_stress import (
    SCENARIOS,
    category_for_feature,
    explain_scenario_outcome,
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
            "Weighted requirement match",
        )
        self.assertNotContains(
            response,
            "Scenario score",
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