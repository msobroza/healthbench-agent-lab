"""Tests for healthbench_agent.llm_eval.runner — EvalRunner.

Tests evaluate_sample, run_async, and tag metrics computation
using mocked samplers. No network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.domain.sampler import SamplerResponse
from healthbench_agent.llm_eval.runner import EvalRunner, _compute_tag_metrics
from tests.conftest import make_sample


def _make_grading_sampler(responses: list[dict]) -> MagicMock:
    """Create a mock sampler that returns grading JSON responses."""
    sampler = MagicMock()
    import json

    side_effects = [
        SamplerResponse(
            response_text=json.dumps(r),
            actual_queried_message_list=[],
            response_metadata={},
        )
        for r in responses
    ]
    sampler.side_effect = side_effects
    return sampler


# ---------------------------------------------------------------------------
# evaluate_sample tests
# ---------------------------------------------------------------------------


class TestEvaluateSample:
    """Tests for EvalRunner.evaluate_sample()."""

    def test_returns_single_eval_result(self):
        sample = make_sample(
            rubric_points=[5.0, -2.0],
            example_tags=["theme:test"],
        )
        sampler = _make_grading_sampler([
            {"explanation": "Good", "criteria_met": True},
            {"explanation": "Bad", "criteria_met": False},
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "Test response")

        assert result.score is not None
        assert result.convo is not None
        assert result.example_level_metadata["prompt_id"] == sample.prompt_id

    def test_all_met_gives_score_one(self):
        sample = make_sample(rubric_points=[5.0, 3.0])
        sampler = _make_grading_sampler([
            {"explanation": "A", "criteria_met": True},
            {"explanation": "B", "criteria_met": True},
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "Perfect response")
        assert result.score == pytest.approx(1.0)

    def test_none_met_gives_score_zero(self):
        sample = make_sample(rubric_points=[5.0, 3.0])
        sampler = _make_grading_sampler([
            {"explanation": "A", "criteria_met": False},
            {"explanation": "B", "criteria_met": False},
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "Bad response")
        assert result.score == pytest.approx(0.0)

    def test_penalty_met_gives_negative_score(self):
        sample = make_sample(rubric_points=[5.0, -10.0])
        sampler = _make_grading_sampler([
            {"explanation": "A", "criteria_met": False},  # 5pt positive not met
            {"explanation": "B", "criteria_met": True},  # -10pt penalty met
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "Harmful response")
        assert result.score is not None
        assert result.score < 0

    def test_conversation_includes_agent_response(self):
        sample = make_sample()
        sampler = _make_grading_sampler([
            {"explanation": "A", "criteria_met": True},
            {"explanation": "B", "criteria_met": True},
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "My response")
        # Last turn should be the agent's response
        last_turn = result.convo[-1]
        assert last_turn["role"] == "assistant"
        assert last_turn["content"] == "My response"

    def test_verdicts_stored_in_metadata(self):
        sample = make_sample(rubric_points=[5.0])
        sampler = _make_grading_sampler([
            {"explanation": "Correct", "criteria_met": True},
        ])
        runner = EvalRunner(sampler=sampler)
        result = runner.evaluate_sample(sample, "Response")
        verdicts = result.example_level_metadata["verdicts"]
        assert len(verdicts) == 1
        assert verdicts[0]["criteria_met"] is True
        assert verdicts[0]["explanation"] == "Correct"


# ---------------------------------------------------------------------------
# run_async tests
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for EvalRunner.run_async() with mocked samplers."""

    def test_returns_results_in_order(self):
        samples = [
            make_sample(prompt_id=f"sample-{i}", rubric_points=[5.0])
            for i in range(3)
        ]
        # Each sample needs one grading response
        sampler = _make_grading_sampler([
            {"explanation": f"Result {i}", "criteria_met": i % 2 == 0}
            for i in range(3)
        ])
        runner = EvalRunner(sampler=sampler, max_workers=2)
        results = runner.run_async(
            samples, ["resp0", "resp1", "resp2"]
        )
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.example_level_metadata["prompt_id"] == f"sample-{i}"

    def test_single_sample(self):
        sample = make_sample(rubric_points=[5.0])
        sampler = _make_grading_sampler([
            {"explanation": "OK", "criteria_met": True},
        ])
        runner = EvalRunner(sampler=sampler, max_workers=1)
        results = runner.run_async([sample], ["response"])
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)

    def test_empty_samples_returns_empty(self):
        sampler = MagicMock()
        runner = EvalRunner(sampler=sampler)
        results = runner.run_async([], [])
        assert results == []


# ---------------------------------------------------------------------------
# run() dispatch tests
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for EvalRunner.run() mode dispatch."""

    def test_async_mode_calls_run_async(self):
        sample = make_sample(rubric_points=[5.0])
        sampler = _make_grading_sampler([
            {"explanation": "OK", "criteria_met": True},
        ])
        runner = EvalRunner(sampler=sampler, mode="async")
        results = runner.run([sample], ["response"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# _compute_tag_metrics tests
# ---------------------------------------------------------------------------


class TestComputeTagMetrics:
    """Tests for _compute_tag_metrics() helper."""

    def test_single_tag_returns_score(self):
        from healthbench_agent.domain.evaluation import CriterionVerdict

        sample = make_sample(rubric_points=[5.0])
        verdicts = [CriterionVerdict("criterion_0", criteria_met=True)]
        metrics = _compute_tag_metrics(sample, verdicts)
        # The rubric item has tags ["level:example", "axis:accuracy"]
        assert "axis:accuracy" in metrics

    def test_empty_rubric_returns_empty_metrics(self):
        from healthbench_agent.domain.evaluation import CriterionVerdict

        sample = make_sample(rubric_points=[])
        sample.rubrics = []
        metrics = _compute_tag_metrics(sample, [])
        assert metrics == {}
