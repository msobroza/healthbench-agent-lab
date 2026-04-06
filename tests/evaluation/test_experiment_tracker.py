"""Tests for evaluation.experiment_tracker — MLflow logging wrapper.

Tests RunParams, RunMetrics, build_run_params, log_evaluation_run, and
log_comparison with a temporary MLflow tracking URI. No remote server required.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from evaluation.experiment_tracker import (
    build_run_metrics,
    build_run_params,
    log_comparison,
    log_evaluation_run,
)
from healthbench_agent.agent import RootAgentPipelineConfig
from healthbench_agent.domain.experiment import RunMetrics, RunParams
from healthbench_agent.llm_eval.config_grader import EvalMode, JudgeConfig

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
        model="gemini-2.5-flash",
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
        assert d["model"] == "gemini-2.5-flash"
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
            grader_model="gemini-2.5-flash",
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

    def test_params_logged_to_mlflow(self, tmp_mlflow, sample_params, sample_metrics):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        run = mlflow.get_run(run_id)
        assert run.data.params["agent_name"] == "baseline_agent"
        assert run.data.params["model"] == "gemini-2.5-flash"

    def test_metrics_logged_to_mlflow(self, tmp_mlflow, sample_params, sample_metrics):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        run = mlflow.get_run(run_id)
        assert run.data.metrics["overall_score"] == pytest.approx(0.65)
        assert run.data.metrics["theme/safety/mean"] == pytest.approx(0.80)

    def test_results_json_artifact_logged(self, tmp_mlflow, sample_params, sample_metrics):
        results = {"sample_0": {"score": 0.8}, "sample_1": {"score": 0.5}}
        run_id = log_evaluation_run(sample_params, sample_metrics, results_json=results)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "results.json" in artifact_names

    def test_prompt_yaml_artifact_logged(self, tmp_mlflow, tmp_path, sample_params, sample_metrics):
        prompt_path = tmp_path / "v1_llm_grader.yaml"
        prompt_path.write_text("version: '1.0.0'\ntemplate: 'test'")
        run_id = log_evaluation_run(
            sample_params,
            sample_metrics,
            prompt_yaml_path=prompt_path,
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "v1_llm_grader.yaml" in artifact_names

    def test_no_artifacts_when_none_provided(self, tmp_mlflow, sample_params, sample_metrics):
        run_id = log_evaluation_run(sample_params, sample_metrics)
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) == 0

    def test_nonexistent_prompt_path_skipped(self, tmp_mlflow, sample_params, sample_metrics):
        run_id = log_evaluation_run(
            sample_params,
            sample_metrics,
            prompt_yaml_path=Path("/nonexistent/grader.yaml"),
        )
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) == 0

    def test_custom_experiment_name(self, tmp_mlflow, sample_params, sample_metrics):
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


# ---------------------------------------------------------------------------
# build_run_params tests
# ---------------------------------------------------------------------------


class TestBuildRunParams:
    """Tests for build_run_params() bridging configs → RunParams."""

    def _agent(self, **overrides) -> RootAgentPipelineConfig:
        defaults = {
            "name": "baseline",
            "model": "gemini-2.5-flash",
            "prompt_version": "1.0.0",
        }
        defaults.update(overrides)
        return RootAgentPipelineConfig(**defaults)

    def test_extracts_grader_provider_from_judge_config(self):
        judge = JudgeConfig(provider="gemini", model="gemini-2.5-flash")
        params = build_run_params(judge, self._agent(), sample_size=50)
        assert params.grader_provider == "gemini"
        assert params.grader_model == "gemini-2.5-flash"

    def test_extracts_agent_fields_from_agent_config(self):
        agent = self._agent(name="tool_agent", model="gemini-2.5-pro", prompt_version="2.0.0")
        params = build_run_params(JudgeConfig(), agent, sample_size=100)
        assert params.agent_name == "tool_agent"
        assert params.model == "gemini-2.5-pro"
        assert params.prompt_version == "2.0.0"

    def test_extracts_eval_mode_as_string(self):
        judge = JudgeConfig(mode=EvalMode.BATCH)
        params = build_run_params(judge, self._agent(), sample_size=100)
        assert params.eval_mode == "batch"

    def test_extracts_grader_temperature(self):
        judge = JudgeConfig(temperature=0.0)
        params = build_run_params(judge, self._agent(), sample_size=50)
        assert params.grader_temperature == 0.0

    def test_loads_prompt_version_and_hash(self):
        judge = JudgeConfig(prompt_path="prompts/llm_grader/v1_llm_grader.yaml")
        params = build_run_params(judge, self._agent(), sample_size=50)
        assert params.grader_prompt_version == "1.0.0"
        assert len(params.grader_prompt_sha256) == 64

    def test_returns_run_params_instance(self):
        params = build_run_params(JudgeConfig(), self._agent(name="multi_agent"), sample_size=200)
        assert isinstance(params, RunParams)
        assert params.agent_name == "multi_agent"
        assert params.sample_size == 200

    def test_seed_stored_in_params(self):
        params = build_run_params(JudgeConfig(), self._agent(), sample_size=50, seed=42)
        assert params.seed == 42

    def test_seed_defaults_to_zero(self):
        params = build_run_params(JudgeConfig(), self._agent(), sample_size=50)
        assert params.seed == 0

    def test_seed_in_to_dict(self):
        params = build_run_params(JudgeConfig(), self._agent(), sample_size=50, seed=123)
        assert params.to_dict()["seed"] == 123


# ---------------------------------------------------------------------------
# build_run_metrics tests
# ---------------------------------------------------------------------------


class TestBuildRunMetrics:
    """Tests for build_run_metrics() aggregation helper."""

    def test_returns_run_metrics_instance(self):
        from healthbench_agent.domain.evaluation import SingleEvalResult

        results = [
            SingleEvalResult(
                score=0.8,
                metrics={"theme:safety": 0.9, "axis:accuracy": 0.7},
            ),
        ]
        metrics = build_run_metrics(results)
        assert isinstance(metrics, RunMetrics)

    def test_overall_score_aggregated(self):
        from healthbench_agent.domain.evaluation import SingleEvalResult

        results = [
            SingleEvalResult(score=1.0, metrics={}),
            SingleEvalResult(score=0.5, metrics={}),
        ]
        metrics = build_run_metrics(results)
        assert metrics.overall_score == pytest.approx(0.75)

    def test_theme_scores_extracted(self):
        from healthbench_agent.domain.evaluation import SingleEvalResult

        results = [
            SingleEvalResult(
                score=0.8,
                metrics={"theme:safety": 0.9, "theme:accuracy": 0.7},
            ),
        ]
        metrics = build_run_metrics(results)
        assert "safety" in metrics.theme_scores
        assert "accuracy" in metrics.theme_scores

    def test_axis_scores_extracted(self):
        from healthbench_agent.domain.evaluation import SingleEvalResult

        results = [
            SingleEvalResult(
                score=0.8,
                metrics={"axis:completeness": 0.85},
            ),
        ]
        metrics = build_run_metrics(results)
        assert "completeness" in metrics.axis_scores

    def test_empty_results_returns_zero(self):
        metrics = build_run_metrics([])
        assert metrics.overall_score == 0.0
        assert metrics.theme_scores == {}
        assert metrics.axis_scores == {}


# ---------------------------------------------------------------------------
# RootAgentPipelineConfig tests
# ---------------------------------------------------------------------------


class TestRootAgentPipelineConfig:
    """Tests for RootAgentPipelineConfig pydantic-settings class."""

    def test_default_values(self):
        config = RootAgentPipelineConfig()
        assert config.name == "baseline_agent"
        assert config.model == "gemini-2.5-flash"
        assert config.prompt_version == "1.0.0"

    def test_from_yaml_loads_agent_config(self):
        config = RootAgentPipelineConfig.from_yaml("config/agents/tool_agent.yaml")
        assert config.name == "tool_agent"
        assert config.prompt_version == "2.0.0"

    def test_from_yaml_overrides_take_priority(self):
        config = RootAgentPipelineConfig.from_yaml(
            "config/agents/baseline_agent.yaml",
            model="gemini-2.5-pro",
        )
        assert config.model == "gemini-2.5-pro"
        assert config.name == "baseline_agent"

    def test_api_keys_default_to_none(self, monkeypatch):
        # Scrub env vars so the test is hermetic regardless of local .env
        for var in (
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "AGENT_OPENAI_API_KEY",
            "AGENT_GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        config = RootAgentPipelineConfig(_env_file=None)
        assert config.google_api_key is None
        assert config.openai_api_key is None

    def test_api_key_masked_in_model_dump(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "secret-key-123")
        config = RootAgentPipelineConfig()
        dump = config.model_dump()
        assert "secret-key-123" not in str(dump)

    def test_description_field_default(self):
        config = RootAgentPipelineConfig()
        assert config.description == "Health assistant agent."

    def test_description_from_yaml(self):
        config = RootAgentPipelineConfig.from_yaml("config/agents/baseline_agent.yaml")
        assert config.description == "Baseline health assistant with no tools or sub-agents."
