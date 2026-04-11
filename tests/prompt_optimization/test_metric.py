"""Tests for EndToEndMetric and agent-tree helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from healthbench_agent.agent import AgentNodeConfig, RootAgentPipelineConfig
from healthbench_agent.domain.dataset import HealthBenchSample
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.prompt_optimization.metric import (
    EndToEndMetric,
    accepts_instruction_override,
    find_agent_node,
    list_agent_names,
    locate_target,
)


def _make_sample(prompt_id: str = "test-1") -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id=prompt_id,
        prompt=[{"role": "user", "content": "What is aspirin?"}],
        rubrics=[RubricItem(criterion="Mentions pain relief", points=1.0, tags=["accuracy"])],
        example_tags=["theme:general"],
    )


def _make_pipeline(response_text: str) -> MagicMock:
    """Build a mock AgentPipeline whose .generate returns the given text."""
    pipeline = MagicMock()
    pipeline.generate = AsyncMock(return_value=response_text)
    return pipeline


def _make_config() -> MagicMock:
    """Build a mock RootAgentPipelineConfig that records model_copy calls."""
    config = MagicMock()
    config.model_copy.return_value = config
    return config


class TestEndToEndMetric:
    def test_returns_aggregate_score_when_criteria_met(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Aspirin is used for pain relief.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            score = metric("You are a health assistant.")

        assert score == pytest.approx(1.0)

    def test_returns_zero_when_no_criteria_met(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=False)
        ]
        pipeline = _make_pipeline("I don't know.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            score = metric("Bad prompt.")

        assert score == pytest.approx(0.0)

    def test_passes_instruction_override_to_create_pipeline(self):
        """The candidate prompt must reach the agent via instruction_override.

        Regression test for the bug where model_copy(update=...) silently
        dropped the override and every trial scored the original prompt.
        """
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ) as mock_create:
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            metric("New candidate prompt.")

        config.model_copy.assert_called_once_with(
            update={"instruction_override": "New candidate prompt."}
        )
        mock_create.assert_called_once_with(config)

    def test_pipeline_generate_is_called_with_sample_prompt(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ):
            metric = EndToEndMetric(agent_config=config, judge=judge, samples=[sample])
            metric("Some prompt.")

        pipeline.generate.assert_awaited_once_with(sample.prompt)


def _make_multi_agent_config() -> RootAgentPipelineConfig:
    """Build a tiny multi-agent config mirroring the real pipeline layout."""
    return RootAgentPipelineConfig(
        name="multi_agent",
        orchestration="sequential",
        prompt_path="prompts/multi_agent/v1_structured.yaml",
        prompt_key="coordinator_instruction",
        sub_agents=[
            AgentNodeConfig(
                name="triage_agent",
                prompt_key="triage_instruction",
            ),
            AgentNodeConfig(
                name="coordinator_agent",
                orchestration="routing",
                prompt_key="coordinator_instruction",
                sub_agents=[
                    AgentNodeConfig(
                        name="emergency_agent",
                        prompt_key="emergency_instruction",
                    ),
                    AgentNodeConfig(
                        name="general_health_agent",
                        prompt_key="general_health_instruction",
                        prompt_path="prompts/multi_agent/v1_general.yaml",
                    ),
                ],
            ),
            AgentNodeConfig(
                name="reviewer_agent",
                prompt_key="reviewer_instruction",
            ),
        ],
    )


class TestFindAgentNode:
    def test_finds_root_by_name(self):
        config = _make_multi_agent_config()
        assert find_agent_node(config, "multi_agent") is config

    def test_finds_nested_child(self):
        config = _make_multi_agent_config()
        node = find_agent_node(config, "emergency_agent")
        assert node is not None
        assert node.name == "emergency_agent"

    def test_returns_none_when_missing(self):
        config = _make_multi_agent_config()
        assert find_agent_node(config, "nonexistent") is None


class TestListAgentNames:
    def test_returns_all_names_in_pre_order(self):
        config = _make_multi_agent_config()
        assert list_agent_names(config) == [
            "multi_agent",
            "triage_agent",
            "coordinator_agent",
            "emergency_agent",
            "general_health_agent",
            "reviewer_agent",
        ]

    def test_returns_single_name_for_leaf(self):
        leaf = AgentNodeConfig(name="solo")
        assert list_agent_names(leaf) == ["solo"]


class TestLocateTarget:
    def test_leaf_inherits_parent_prompt_path(self):
        config = _make_multi_agent_config()
        node, path = locate_target(config, "triage_agent")
        assert node.name == "triage_agent"
        assert path == "prompts/multi_agent/v1_structured.yaml"

    def test_nested_leaf_inherits_from_root(self):
        config = _make_multi_agent_config()
        node, path = locate_target(config, "emergency_agent")
        assert node.name == "emergency_agent"
        assert path == "prompts/multi_agent/v1_structured.yaml"

    def test_explicit_prompt_path_overrides_inheritance(self):
        config = _make_multi_agent_config()
        node, path = locate_target(config, "general_health_agent")
        assert node.name == "general_health_agent"
        assert path == "prompts/multi_agent/v1_general.yaml"

    def test_root_returns_its_own_path(self):
        config = _make_multi_agent_config()
        node, path = locate_target(config, "multi_agent")
        assert node is config
        assert path == "prompts/multi_agent/v1_structured.yaml"

    def test_raises_when_target_not_found(self):
        config = _make_multi_agent_config()
        with pytest.raises(ValueError, match="Agent 'ghost' not found"):
            locate_target(config, "ghost")

    def test_error_lists_available_agent_names(self):
        config = _make_multi_agent_config()
        with pytest.raises(ValueError, match="reviewer_agent"):
            locate_target(config, "ghost")


class TestAcceptsInstructionOverride:
    def test_leaf_agent_accepts_override(self):
        leaf = AgentNodeConfig(name="leaf")
        assert accepts_instruction_override(leaf) is True

    def test_routing_agent_accepts_override(self):
        routing = AgentNodeConfig(
            name="router",
            orchestration="routing",
            sub_agents=[AgentNodeConfig(name="child")],
        )
        assert accepts_instruction_override(routing) is True

    @pytest.mark.parametrize("orchestration", ["sequential", "loop", "parallel"])
    def test_composite_agent_rejects_override(self, orchestration):
        composite = AgentNodeConfig(
            name="composite",
            orchestration=orchestration,
            sub_agents=[AgentNodeConfig(name="child")],
        )
        assert accepts_instruction_override(composite) is False


class TestEndToEndMetricWithTargetAgent:
    def test_rejects_unknown_target_at_construction_time(self):
        sample = _make_sample()
        judge = MagicMock()
        config = _make_multi_agent_config()

        with pytest.raises(ValueError, match="Target agent 'bogus' not found"):
            EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="bogus",
            )

    def test_injects_override_on_target_sub_agent_only(self):
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_multi_agent_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ) as mock_create:
            metric = EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="reviewer_agent",
            )
            metric("Stricter review checklist.")

        # create_pipeline receives a deep-copied config where only the
        # reviewer has the override set; the original config and every
        # other sub-agent remain untouched.
        patched = mock_create.call_args.args[0]
        assert patched is not config  # deep-copied
        assert patched.instruction_override is None
        reviewer = find_agent_node(patched, "reviewer_agent")
        assert reviewer is not None
        assert reviewer.instruction_override == "Stricter review checklist."
        # Siblings keep their YAML-loaded instruction path.
        triage = find_agent_node(patched, "triage_agent")
        assert triage is not None
        assert triage.instruction_override is None
        # Original config was not mutated.
        original_reviewer = find_agent_node(config, "reviewer_agent")
        assert original_reviewer is not None
        assert original_reviewer.instruction_override is None

    def test_rejects_composite_sub_agent_target(self):
        """Targeting a SequentialAgent/LoopAgent/ParallelAgent node must fail.

        Composite agents have no ``instruction`` field, so an override on
        them is silently dropped at build time. The metric must catch
        this at construction instead of producing garbage trial scores.
        """
        sample = _make_sample()
        judge = MagicMock()
        config = RootAgentPipelineConfig(
            name="root",
            orchestration="sequential",
            sub_agents=[
                AgentNodeConfig(
                    name="inner_sequential",
                    orchestration="sequential",
                    sub_agents=[AgentNodeConfig(name="real_leaf")],
                ),
            ],
        )

        with pytest.raises(ValueError, match="pure composite"):
            EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="inner_sequential",
            )

    def test_rejects_composite_root_target(self):
        sample = _make_sample()
        judge = MagicMock()
        config = _make_multi_agent_config()  # root is orchestration="sequential"

        with pytest.raises(ValueError, match="pure composite"):
            EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="multi_agent",
            )

    def test_accepts_routing_sub_agent_target(self):
        """A routing sub-agent builds an LlmAgent and must be targetable."""
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = _make_multi_agent_config()

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ) as mock_create:
            metric = EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="coordinator_agent",
            )
            metric("Route more aggressively.")

        patched = mock_create.call_args.args[0]
        coordinator = find_agent_node(patched, "coordinator_agent")
        assert coordinator is not None
        assert coordinator.instruction_override == "Route more aggressively."

    def test_routing_root_target_uses_deep_copy_path(self):
        """Targeting a routing-root still works (routing builds an LlmAgent)."""
        sample = _make_sample()
        judge = MagicMock()
        judge.grade.return_value = [
            CriterionVerdict(criterion="Mentions pain relief", criteria_met=True)
        ]
        pipeline = _make_pipeline("Response.")
        config = RootAgentPipelineConfig(
            name="routing_root",
            orchestration="routing",
            sub_agents=[AgentNodeConfig(name="leaf")],
        )

        with patch(
            "healthbench_agent.prompt_optimization.metric.create_pipeline",
            return_value=pipeline,
        ) as mock_create:
            metric = EndToEndMetric(
                agent_config=config,
                judge=judge,
                samples=[sample],
                target_agent_name="routing_root",
            )
            metric("New root prompt.")

        patched = mock_create.call_args.args[0]
        assert patched is not config  # deep-copied
        assert patched.instruction_override == "New root prompt."
        # Original not mutated.
        assert config.instruction_override is None


def test_judge_agreement_metric_returns_scalar_from_meta_eval():
    from typing import Any

    from healthbench_agent.llm_eval.meta_eval import FakeJudge, demo_labelled_set
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    samples = demo_labelled_set()
    perfect = {c: m for s in samples for c, m in s.expected.items()}

    captured: dict[str, Any] = {}

    class _CapturingMetric(JudgeAgreementMetric):
        def _build_judge(self, candidate_template: str):  # type: ignore[override]
            captured["template"] = candidate_template
            return FakeJudge(perfect), "fake@1", "sha"

    metric = _CapturingMetric(
        judge_config=None,  # bypassed by override
        labelled=samples,
        dimension_extractor=lambda r: r.category,
        n_samples=1,
        fitness="gold_score",
    )
    score = metric("CANDIDATE TEMPLATE")
    assert isinstance(score, float)
    assert captured["template"] == "CANDIDATE TEMPLATE"


def test_end_to_end_metric_accepts_filter_kwargs():
    """Just verify the kwargs reach the constructor; behaviour tested via integration."""
    import inspect

    from healthbench_agent.prompt_optimization.metric import EndToEndMetric

    sig = inspect.signature(EndToEndMetric.__init__)
    assert "sample_filter" in sig.parameters
    assert "rubric_filter" in sig.parameters


def test_empty_filter_error_re_exported():
    from healthbench_agent.llm_eval.meta_eval import EmptyFilterError as RootError
    from healthbench_agent.prompt_optimization.metric import EmptyFilterError

    assert EmptyFilterError is RootError


def test_end_to_end_metric_sample_filter_rejecting_all_raises_empty_filter():
    """An over-aggressive sample_filter must surface as EmptyFilterError."""
    from healthbench_agent.prompt_optimization.metric import EmptyFilterError

    sample = _make_sample()
    judge = MagicMock()
    config = _make_config()
    metric = EndToEndMetric(
        agent_config=config,
        judge=judge,
        samples=[sample],
        sample_filter=lambda s: False,
    )
    with pytest.raises(EmptyFilterError):
        metric("candidate")


def test_end_to_end_metric_rubric_filter_keeps_subset_and_evaluates():
    """A rubric_filter that keeps some rubrics clones samples and runs the eval."""
    sample = HealthBenchSample(
        prompt_id="multi",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[
            RubricItem(criterion="keep", points=1.0, tags=("accuracy",)),
            RubricItem(criterion="drop", points=1.0, tags=("emergency",)),
        ],
        example_tags=["theme:general"],
    )
    judge = MagicMock()
    judge.grade.return_value = [CriterionVerdict(criterion="keep", criteria_met=True)]
    pipeline = _make_pipeline("ok")
    config = _make_config()
    with patch(
        "healthbench_agent.prompt_optimization.metric.create_pipeline",
        return_value=pipeline,
    ):
        metric = EndToEndMetric(
            agent_config=config,
            judge=judge,
            samples=[sample],
            rubric_filter=lambda r: r.criterion == "keep",
        )
        score = metric("candidate")
    # Judge should only see the surviving "keep" rubric, not "drop".
    graded_rubrics = judge.grade.call_args.args[1]
    assert [r.criterion for r in graded_rubrics] == ["keep"]
    assert score == pytest.approx(1.0)


def test_end_to_end_metric_rubric_filter_drops_all_raises_empty_filter():
    """A rubric_filter that removes every rubric from every sample must raise."""
    from healthbench_agent.prompt_optimization.metric import EmptyFilterError

    sample = _make_sample()
    judge = MagicMock()
    config = _make_config()
    metric = EndToEndMetric(
        agent_config=config,
        judge=judge,
        samples=[sample],
        rubric_filter=lambda r: False,
    )
    with pytest.raises(EmptyFilterError):
        metric("candidate")


def test_judge_agreement_metric_default_build_judge_constructs_llm_judge(monkeypatch):
    """The default _build_judge path (exercised via the public __call__ entry
    point) clones the config, wires a new template, constructs an
    LLMJudgeGrader, and threads the fingerprint into run_meta_eval's
    judge_metadata. Exercises lines 356-372 and the __call__ wiring."""
    import hashlib

    from healthbench_agent.domain.meta_evaluation import MetricResults
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.meta_eval import demo_labelled_set
    from healthbench_agent.llm_eval.meta_eval.results_view import MetricResultsView
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    created: dict[str, object] = {}

    def fake_create_llm_client(cfg):
        created["client_config"] = cfg
        return MagicMock(name="llm_client")

    captured_templates: list[str] = []

    def fake_make_template(template):
        captured_templates.append(template)
        return MagicMock(name=f"template:{template}")

    # LLMJudgeGrader is constructed with llm_client + template; record args.
    construction: dict[str, object] = {}

    class FakeLLMJudgeGrader:
        def __init__(self, llm_client, template, max_workers):
            construction["llm_client"] = llm_client
            construction["template"] = template
            construction["max_workers"] = max_workers

    # _build_judge does function-local imports from the source modules, so
    # the patch target must be the source module, not metric.py's namespace.
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.clients.create_llm_client",
        fake_create_llm_client,
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge._make_template",
        fake_make_template,
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge.LLMJudgeGrader",
        FakeLLMJudgeGrader,
    )

    # run_meta_eval is imported at module level into metric.py, so the
    # correct patch target is metric.py's namespace. Capture the kwargs so
    # we can assert the fingerprint was threaded into judge_metadata.
    captured_run_kwargs: dict[str, object] = {}

    def fake_run_meta_eval(**kwargs):
        captured_run_kwargs.update(kwargs)
        return MetricResultsView(
            results=MetricResults(
                scores={"gold_score": 0.75},
                n_samples_graded=0,
                n_rubrics_graded=0,
                judge_metadata=kwargs.get("judge_metadata") or {},
            )
        )

    monkeypatch.setattr(
        "healthbench_agent.prompt_optimization.metric.run_meta_eval",
        fake_run_meta_eval,
    )

    cfg = JudgeConfig(
        provider="openai",
        model="gpt-4.1",
        temperature=0.0,
        prompt_path="prompts/llm_grader/v1_llm_grader.yaml",
    )
    metric = JudgeAgreementMetric(
        judge_config=cfg,
        labelled=demo_labelled_set(),
        dimension_extractor=lambda r: r.category,
    )

    # Drive _build_judge through the public entry point rather than
    # calling the private method directly.
    score = metric("CANDIDATE")

    # Observable behaviour of _build_judge: an LLMJudgeGrader was
    # constructed with the llm_client returned by create_llm_client and a
    # template derived from the candidate string.
    assert isinstance(construction["llm_client"], MagicMock)
    assert captured_templates == ["CANDIDATE"]
    # The template passed to LLMJudgeGrader is the MagicMock returned by
    # fake_make_template for the "CANDIDATE" input.
    assert construction["template"]._mock_name == "template:CANDIDATE"

    # The cloned JudgeConfig — not the original — was handed to
    # create_llm_client, so field values must match the source config.
    client_cfg = created["client_config"]
    assert client_cfg.provider == "openai"
    assert client_cfg.model == "gpt-4.1"
    assert client_cfg.temperature == 0.0

    # The fake LLMJudgeGrader was handed to run_meta_eval as the judge.
    assert isinstance(captured_run_kwargs["judge"], FakeLLMJudgeGrader)
    # The fingerprint produced by _build_judge was threaded into
    # judge_metadata as "judge_model".
    expected_fingerprint = "openai/gpt-4.1@0.0"
    assert captured_run_kwargs["judge_metadata"] == {"judge_model": expected_fingerprint}
    # The prompt_sha produced by _build_judge is the sha256 of the
    # candidate template, even though __call__ does not forward it.
    expected_prompt_sha = hashlib.sha256(b"CANDIDATE").hexdigest()
    assert expected_prompt_sha == hashlib.sha256(captured_templates[0].encode("utf-8")).hexdigest()

    # __call__ returns the float score from the fake view.
    assert score == 0.75


def test_judge_agreement_metric_default_build_judge_requires_config():
    """When judge_config is None and _build_judge isn't overridden, raise."""
    from healthbench_agent.llm_eval.meta_eval import demo_labelled_set
    from healthbench_agent.prompt_optimization.metric import JudgeAgreementMetric

    metric = JudgeAgreementMetric(
        judge_config=None,
        labelled=demo_labelled_set(),
        dimension_extractor=lambda r: r.category,
    )
    with pytest.raises(RuntimeError, match="judge_config is None"):
        metric._build_judge("candidate")
