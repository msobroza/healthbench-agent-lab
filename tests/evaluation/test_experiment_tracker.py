"""Tests for evaluation.experiment_tracker — MLflow logging wrapper.

Tests RunParams, RunMetrics, log_evaluation_run, and log_comparison
with a temporary MLflow tracking URI. No remote server required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest

from evaluation.experiment_tracker import (
    EXPERIMENT_NAME,
    RunMetrics,
    RunParams,
    log_comparison,
    log_evaluation_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_mlflow(tmp_path):
    """Set MLflow tracking to a temp directory for test isolation."""
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tmp_path
    mlflow.set_tracking_uri("")


@pytest.fixture()
def sample_params():
    """Minimal RunParams for testing."""
    return RunParams(
        agent_name="baseline_agent",
        prompt_version="1.0.0",
        model="gemini-2.0-flash",
        sample_size=50,
    )


@pytest.fixture()
def sample_metrics():
    """Minimal RunMetrics for testing."""
    return RunMetrics(
        overall_score=0.65,
        theme_scores={"safety": 0.80, "accuracy": 0.55},
        axis_scores={"completeness": 0.70},
    )


# ---------------------------------------------------------------------------
# RunParams tests
# ---------------------------------------------------------------------------


class TestRunParams:
    """Tests for RunParams dataclass."""

    def test_to_dict_contains_required_keys(self, sample_params):
        d = sample_params.to_dict()
        assert d["agent_name"] == "baseline_agent"
        assert d["prompt_version"] == "1.0.0"
        assert d["model"] == "gemini-2.0-flash"
        assert d["sample_size"] == 50
        assert "timestamp" in d

    def test_to_dict_contains_grader_keys(self, sample_params):
        d = sample_params.to_dict()
        assert d["grader_provider"] == "openai"
        assert d["grader_model"] == "gpt-4.1-2025-04-14"
        assert d["grader_temperature"] == 0.0
        assert d["grader_prompt_version"] == "1.0.0"
        assert "grader_prompt_sha256" in d
        assert d["eval_mode"] == "async"

    def test_to_dict_timestamp_is_iso_format(self, sample_params):
        d = sample_params.to_dict()
        # ISO format includes 'T' separator
        assert "T" in d["timestamp"]

    def test_custom_grader_params(self):
        params = RunParams(
            agent_name="tool_agent",
            prompt_version="2.0.0",
            model="gemini-2.5-pro",
            sample_size=100,
            grader_provider="gemini",
            grader_model="gemini-2.0-flash",
            grader_temperature=0.0,
            eval_mode="batch",
        )
        d = params.to_dict()
        assert d["grader_provider"] == "gemini"
        assert d["eval_mode"] == "batch"


# ---------------------------------------------------------------------------
# RunMetrics tests
# ---------------------------------------------------------------------------


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_to_flat_dict_contains_overall_score(self, sample_metrics):
        d = sample_metrics.to_flat_dict()
        assert d["overall_score"] == 0.65

    def test_to_flat_dict_prefixes_theme_scores(self, sample_metrics):
        d = sample_metrics.to_flat_dict()
        assert d["theme/safety/mean"] == 0.80
        assert d["theme/accuracy/mean"] == 0.55

    def test_to_flat_dict_prefixes_axis_scores(self, sample_metrics):
        d = sample_metrics.to_flat_dict()
        assert d["axis/completeness/mean"] == 0.70

    def test_empty_theme_and_axis_scores(self):
        metrics = RunMetrics(overall_score=0.5)
        d = metrics.to_flat_dict()
        assert d == {"overall_score": 0.5}

    def test_many_dimensions(self):
        metrics = RunMetrics(
            overall_score=0.72,
            theme_scores={
                "safety": 0.80,
                "accuracy": 0.65,
                "communication": 0.70,
            },
            axis_scores={
                "completeness": 0.75,
                "harm_avoidance": 0.90,
            },
        )
        d = metrics.to_flat_dict()
        assert len(d) == 6  # 1 overall + 3 theme + 2 axis


# ---------------------------------------------------------------------------
# log_evaluation_run tests
# ---------------------------------------------------------------------------


class TestLogEvaluationRun:
    """Tests for log_evaluation_run() with temp MLflow backend."""

    def test_returns_run_id(self, tmp_mlflow, sample_params, sample_metrics):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # MLflow run IDs are 32 hex chars

    def test_params_logged_to_mlflow(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        run = mlflow.get_run(run_id)
        assert run.data.params["agent_name"] == "baseline_agent"
        assert run.data.params["model"] == "gemini-2.0-flash"

    def test_metrics_logged_to_mlflow(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        run = mlflow.get_run(run_id)
        assert run.data.metrics["overall_score"] == pytest.approx(0.65)
        assert run.data.metrics["theme/safety/mean"] == pytest.approx(0.80)

    def test_results_json_artifact_logged(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        results = {"sample_0": {"score": 0.8}, "sample_1": {"score": 0.5}}
        run_id = log_evaluation_run(
            sample_params, sample_metrics, results_json=results
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "results.json" in artifact_names

    def test_prompt_yaml_artifact_logged(
        self, tmp_mlflow, tmp_path, sample_params, sample_metrics
    ):
        prompt_path = tmp_path / "grader_v1.yaml"
        prompt_path.write_text("version: '1.0.0'\ntemplate: 'test'")
        run_id = log_evaluation_run(
            sample_params,
            sample_metrics,
            prompt_yaml_path=prompt_path,
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "grader_v1.yaml" in artifact_names

    def test_no_artifacts_when_none_provided(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) == 0

    def test_nonexistent_prompt_path_skipped(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        run_id = log_evaluation_run(
            sample_params,
            sample_metrics,
            prompt_yaml_path=Path("/nonexistent/grader.yaml"),
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) == 0

    def test_custom_experiment_name(
        self, tmp_mlflow, sample_params, sample_metrics
    ):
        run_id = log_evaluation_run(
            sample_params,
            sample_metrics,
            experiment_name="custom-experiment",
        )
        run = mlflow.get_run(run_id)
        experiment = mlflow.get_experiment(run.info.experiment_id)
        assert experiment.name == "custom-experiment"


# ---------------------------------------------------------------------------
# log_comparison tests
# ---------------------------------------------------------------------------


class TestLogComparison:
    """Tests for log_comparison() with temp MLflow backend."""

    def test_returns_run_id(self, tmp_mlflow):
        run_id = log_comparison(
            run_id_a="abc123",
            run_id_b="def456",
            bootstrap_ci={
                "mean_difference": 0.15,
                "ci_lower": 0.05,
                "ci_upper": 0.25,
            },
            t_test={"t_statistic": 3.2, "p_value": 0.003},
            cohens_d=0.65,
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 32

    def test_comparison_params_logged(self, tmp_mlflow):
        run_id = log_comparison(
            run_id_a="aaa",
            run_id_b="bbb",
            bootstrap_ci={"mean_difference": 0.1, "ci_lower": 0.0, "ci_upper": 0.2},
            t_test={"t_statistic": 2.0, "p_value": 0.05},
            cohens_d=0.5,
        )
        run = mlflow.get_run(run_id)
        assert run.data.params["comparison_type"] == "paired"
        assert run.data.params["run_id_a"] == "aaa"
        assert run.data.params["run_id_b"] == "bbb"

    def test_comparison_metrics_logged(self, tmp_mlflow):
        run_id = log_comparison(
            run_id_a="aaa",
            run_id_b="bbb",
            bootstrap_ci={"mean_difference": 0.15, "ci_lower": 0.05, "ci_upper": 0.25},
            t_test={"t_statistic": 3.2, "p_value": 0.003},
            cohens_d=0.65,
        )
        run = mlflow.get_run(run_id)
        assert run.data.metrics["mean_difference"] == pytest.approx(0.15)
        assert run.data.metrics["ci_lower"] == pytest.approx(0.05)
        assert run.data.metrics["ci_upper"] == pytest.approx(0.25)
        assert run.data.metrics["t_statistic"] == pytest.approx(3.2)
        assert run.data.metrics["p_value"] == pytest.approx(0.003)
        assert run.data.metrics["cohens_d"] == pytest.approx(0.65)

    def test_comparison_artifact_logged(self, tmp_mlflow):
        run_id = log_comparison(
            run_id_a="aaa",
            run_id_b="bbb",
            bootstrap_ci={"mean_difference": 0.1, "ci_lower": 0.0, "ci_upper": 0.2},
            t_test={"t_statistic": 2.0, "p_value": 0.05},
            cohens_d=0.5,
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "comparison.json" in artifact_names
