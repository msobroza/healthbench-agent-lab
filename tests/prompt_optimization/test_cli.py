"""Tests for optimize-prompt CLI dispatch (agent vs judge domain)."""

from __future__ import annotations


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
