"""Tests for optimize-prompt CLI dispatch (agent vs judge domain)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


def test_optimize_prompt_cli_judge_domain_writes_to_llm_grader(tmp_path, monkeypatch):
    """--prompt-domain judge writes to <grader_dir>/v2_optimized.yaml."""
    from healthbench_agent.prompt_optimization import cli as optimize_cli

    grader_dir = tmp_path / "prompts" / "llm_grader"
    grader_dir.mkdir(parents=True)
    grader_yaml = grader_dir / "v1_llm_grader.yaml"
    grader_yaml.write_text("version: 1.0.0\ntemplate: 'old template'\n")

    monkeypatch.setattr(
        "healthbench_agent.prompt_optimization.cli._run_judge_optimization",
        lambda args: {
            "optimized_prompt": "NEW TEMPLATE",
            "baseline_score": 0.5,
            "optimized_score": 0.7,
            "improvement": 0.2,
            "num_trials": 3,
            "optimizer_name": "critique_refine",
            "trials": [],
            "target_prompt_path": str(grader_yaml),
        },
    )

    optimize_cli.main_argv(
        [
            "--prompt-domain",
            "judge",
            "--judge-config",
            str(grader_yaml),
            "--optimizer",
            "critique_refine",
            "--sample-size",
            "1",
            "--max-trials",
            "1",
        ]
    )

    out = grader_dir / "v2_optimized.yaml"
    assert out.exists()
    assert "NEW TEMPLATE" in out.read_text()


def test_optimize_prompt_cli_agent_domain_requires_agent_config(capsys):
    """--prompt-domain agent (the default) requires --agent-config."""
    from healthbench_agent.prompt_optimization import cli as optimize_cli

    with pytest.raises(SystemExit) as excinfo:
        optimize_cli.main_argv(["--optimizer", "critique_refine"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--agent-config is required" in err


def test_optimize_prompt_cli_judge_domain_requires_judge_config(capsys):
    """--prompt-domain judge requires --judge-config."""
    from healthbench_agent.prompt_optimization import cli as optimize_cli

    with pytest.raises(SystemExit) as excinfo:
        optimize_cli.main_argv(["--prompt-domain", "judge", "--optimizer", "critique_refine"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--judge-config is required" in err


def test_run_judge_optimization_invokes_optimizer_with_template(tmp_path, monkeypatch):
    """_run_judge_optimization should feed the raw grader template to the optimizer."""
    from healthbench_agent.prompt_optimization import cli as optimize_cli
    from healthbench_agent.prompt_optimization.optimizer import (
        OptimizationResult,
        TrialRecord,
    )

    grader_yaml = tmp_path / "grader.yaml"
    grader_yaml.write_text("template: |\n  You are a grader.\n  Respond with JSON.\n")

    judge_yaml = tmp_path / "judge.yaml"
    judge_yaml.write_text(f"provider: openai\nmodel: x\nprompt_path: {grader_yaml}\n")

    # Build a JudgeConfig that points at the grader YAML so the raw-template
    # read path is exercised against a real file on disk.
    from healthbench_agent.llm_eval.grading.config import JudgeConfig

    fake_cfg = JudgeConfig(provider="openai", model="x", prompt_path=str(grader_yaml))

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.config.JudgeConfig.from_yaml",
        classmethod(lambda cls, path: fake_cfg),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: ([], lambda r: r.category),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_filters",
        lambda args: (None, None),
    )

    captured: dict = {}

    class FakeMetric:
        def __init__(self, **kwargs):
            captured["metric_kwargs"] = kwargs

        def __call__(self, prompt):
            return 0.0

    monkeypatch.setattr(
        "healthbench_agent.prompt_optimization.metric.JudgeAgreementMetric",
        FakeMetric,
    )

    class FakeOptimizer:
        def optimize(self, *, current_prompt, samples, metric):
            captured["current_prompt"] = current_prompt
            captured["samples"] = samples
            return OptimizationResult(
                optimized_prompt="NEW",
                baseline_score=0.5,
                optimized_score=0.75,
                improvement=0.25,
                num_trials=2,
                optimizer_name="critique_refine",
                trial_history=[
                    TrialRecord(trial_id=0, prompt="a", score=0.5, timestamp="t0"),
                    TrialRecord(trial_id=1, prompt="NEW", score=0.75, timestamp="t1"),
                ],
                config={},
            )

    monkeypatch.setattr(
        "healthbench_agent.prompt_optimization.create_prompt_optimizer",
        lambda cfg: FakeOptimizer(),
    )

    args = argparse.Namespace(
        judge_config=str(judge_yaml),
        sample_size=1,
        seed=0,
        optimizer="critique_refine",
        max_trials=2,
        fitness="gold_score",
        rubric_axis=[],
        metadata=[],
    )
    result = optimize_cli._run_judge_optimization(args)

    assert result["optimized_prompt"] == "NEW"
    assert result["baseline_score"] == 0.5
    assert result["optimized_score"] == 0.75
    assert result["num_trials"] == 2
    assert result["target_prompt_path"] == str(grader_yaml)
    # The optimizer must receive the raw template string, not a compiled
    # Jinja2 Template or a YAML dict.
    assert captured["current_prompt"].startswith("You are a grader.")
    # The labelled set is carried on the metric, so samples must be None.
    assert captured["samples"] is None


def test_judge_save_path_returns_sibling_v2_yaml(tmp_path):
    """_judge_save_path puts the output next to the source config."""
    from healthbench_agent.prompt_optimization.cli import _judge_save_path

    src = tmp_path / "judge.yaml"
    out = _judge_save_path(str(src))
    assert out == tmp_path / "v2_optimized.yaml"


def test_save_judge_yaml_writes_payload(tmp_path):
    """_save_judge_yaml should render the standard optimized-prompt YAML payload."""
    import yaml

    from healthbench_agent.prompt_optimization.cli import _save_judge_yaml

    src = tmp_path / "judge.yaml"
    src.write_text("provider: openai\n")
    args = SimpleNamespace(judge_config=str(src))
    out_path = _save_judge_yaml(
        args,
        {
            "optimized_prompt": "new template",
            "baseline_score": 0.5,
            "optimized_score": 0.75,
            "improvement": 0.25,
            "num_trials": 3,
            "optimizer_name": "critique_refine",
            "target_prompt_path": "prompts/llm_grader/v1.yaml",
        },
    )
    assert out_path == tmp_path / "v2_optimized.yaml"
    payload = yaml.safe_load(out_path.read_text())
    assert payload["template"] == "new template"
    assert payload["parent_prompt_path"] == "prompts/llm_grader/v1.yaml"
    assert "critique_refine" in payload["rationale"]
    assert "0.2500" in payload["rationale"]
