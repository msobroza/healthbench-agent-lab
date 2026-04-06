"""Tests for healthbench_agent.llm_eval.runner — EvalRunner.

Tests evaluate_sample, run_async, evaluate_pipeline, and tag metrics
computation using mocked judges. No network calls.
"""

from __future__ import annotations

import asyncio

import pytest

from healthbench_agent.agent import AgentPipeline
from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.config_grader import EvalMode, JudgeConfig
from healthbench_agent.llm_eval.runner import EvalRunner, _compute_tag_metrics
from tests.conftest import make_sample


class _FakeJudge(JudgeGrader):
    """Fake judge that returns predetermined verdicts for testing."""

    def __init__(self, verdicts_per_call: list[list[dict]]) -> None:
        self._verdicts = iter(verdicts_per_call)

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        verdict_dicts = next(self._verdicts)
        return [
            CriterionVerdict(
                criterion=item.criterion,
                criteria_met=v["criteria_met"],
                explanation=v.get("explanation", ""),
            )
            for item, v in zip(rubric_items, verdict_dicts)
        ]


def _make_config(**overrides) -> JudgeConfig:
    """Create a JudgeConfig pointing to the real grader prompt."""
    defaults = {"prompt_path": "prompts/llm_grader/v1_llm_grader.yaml"}
    defaults.update(overrides)
    return JudgeConfig(**defaults)


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
        judge = _FakeJudge(
            [
                [
                    {"explanation": "Good", "criteria_met": True},
                    {"explanation": "Bad", "criteria_met": False},
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "Test response")

        assert result.score is not None
        assert result.convo is not None
        assert result.example_level_metadata["prompt_id"] == sample.prompt_id

    def test_all_met_gives_score_one(self):
        sample = make_sample(rubric_points=[5.0, 3.0])
        judge = _FakeJudge(
            [
                [
                    {"criteria_met": True},
                    {"criteria_met": True},
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "Perfect response")
        assert result.score == pytest.approx(1.0)

    def test_none_met_gives_score_zero(self):
        sample = make_sample(rubric_points=[5.0, 3.0])
        judge = _FakeJudge(
            [
                [
                    {"criteria_met": False},
                    {"criteria_met": False},
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "Bad response")
        assert result.score == pytest.approx(0.0)

    def test_penalty_met_gives_negative_score(self):
        sample = make_sample(rubric_points=[5.0, -10.0])
        judge = _FakeJudge(
            [
                [
                    {"criteria_met": False},  # 5pt positive not met
                    {"criteria_met": True},  # -10pt penalty met
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "Harmful response")
        assert result.score is not None
        assert result.score < 0

    def test_conversation_includes_agent_response(self):
        sample = make_sample()
        judge = _FakeJudge(
            [
                [
                    {"criteria_met": True},
                    {"criteria_met": True},
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "My response")
        last_turn = result.convo[-1]
        assert last_turn["role"] == "assistant"
        assert last_turn["content"] == "My response"

    def test_verdicts_stored_in_metadata(self):
        sample = make_sample(rubric_points=[5.0])
        judge = _FakeJudge(
            [
                [
                    {"explanation": "Correct", "criteria_met": True},
                ]
            ]
        )
        runner = EvalRunner(judge=judge, config=_make_config())
        result = runner.evaluate_sample(sample, "Response")
        verdicts = result.example_level_metadata["verdicts"]
        assert len(verdicts) == 1
        assert verdicts[0]["criteria_met"] is True
        assert verdicts[0]["explanation"] == "Correct"


# ---------------------------------------------------------------------------
# run_async tests
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for EvalRunner.run_async() with fake judges."""

    def test_returns_results_in_order(self):
        samples = [make_sample(prompt_id=f"sample-{i}", rubric_points=[5.0]) for i in range(3)]
        # Each sample gets one call with one verdict
        judge = _FakeJudge([[{"criteria_met": i % 2 == 0}] for i in range(3)])
        runner = EvalRunner(judge=judge, config=_make_config(max_workers=2))
        results = runner.run_async(samples, ["resp0", "resp1", "resp2"])
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.example_level_metadata["prompt_id"] == f"sample-{i}"

    def test_single_sample(self):
        sample = make_sample(rubric_points=[5.0])
        judge = _FakeJudge([[{"criteria_met": True}]])
        runner = EvalRunner(judge=judge, config=_make_config(max_workers=1))
        results = runner.run_async([sample], ["response"])
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)

    def test_empty_samples_returns_empty(self):
        judge = _FakeJudge([])
        runner = EvalRunner(judge=judge, config=_make_config())
        results = runner.run_async([], [])
        assert results == []


# ---------------------------------------------------------------------------
# run() dispatch tests
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for EvalRunner.run() mode dispatch."""

    def test_async_mode_calls_run_async(self):
        sample = make_sample(rubric_points=[5.0])
        judge = _FakeJudge([[{"criteria_met": True}]])
        runner = EvalRunner(
            judge=judge,
            config=_make_config(mode=EvalMode.ASYNC),
        )
        results = runner.run([sample], ["response"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# from_config factory tests
# ---------------------------------------------------------------------------


class TestFromConfig:
    """Tests for EvalRunner.from_config() factory method."""

    def test_creates_runner_with_judge(self):
        config = _make_config()
        runner = EvalRunner.from_config(config)
        # The runner should have a judge attribute (LLMJudgeGrader)
        assert runner.judge is not None
        assert runner.config is config


# ---------------------------------------------------------------------------
# evaluate_pipeline tests
# ---------------------------------------------------------------------------


class _FakeAgentPipeline(AgentPipeline):
    """Fake agent pipeline that echoes the last user message."""

    async def generate(self, conversation: MessageList) -> str:
        last_user = [m for m in conversation if m["role"] == "user"][-1]
        return f"Response to: {last_user['content']}"


class TestEvaluatePipeline:
    """Tests for EvalRunner.evaluate_pipeline() with a fake agent."""

    def test_generates_and_evaluates(self):
        sample = make_sample(rubric_points=[5.0])
        judge = _FakeJudge([[{"criteria_met": True}]])
        runner = EvalRunner(judge=judge, config=_make_config())
        pipeline = _FakeAgentPipeline()
        results = asyncio.get_event_loop().run_until_complete(
            runner.evaluate_pipeline(pipeline, [sample])
        )
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0)
        last_turn = results[0].convo[-1]
        assert last_turn["role"] == "assistant"
        assert "Response to:" in last_turn["content"]

    def test_multiple_samples(self):
        samples = [make_sample(prompt_id=f"p-{i}", rubric_points=[5.0]) for i in range(3)]
        judge = _FakeJudge([[{"criteria_met": True}] for _ in range(3)])
        runner = EvalRunner(judge=judge, config=_make_config(max_workers=2))
        pipeline = _FakeAgentPipeline()
        results = asyncio.get_event_loop().run_until_complete(
            runner.evaluate_pipeline(pipeline, samples)
        )
        assert len(results) == 3
        for r in results:
            assert r.score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_tag_metrics tests
# ---------------------------------------------------------------------------


class TestComputeTagMetrics:
    """Tests for _compute_tag_metrics() helper."""

    def test_single_tag_returns_score(self):
        sample = make_sample(rubric_points=[5.0])
        verdicts = [CriterionVerdict("criterion_0", criteria_met=True)]
        metrics = _compute_tag_metrics(sample, verdicts)
        assert "axis:accuracy" in metrics

    def test_empty_rubric_returns_empty_metrics(self):
        sample = make_sample(rubric_points=[])
        sample.rubrics = []
        metrics = _compute_tag_metrics(sample, [])
        assert metrics == {}
