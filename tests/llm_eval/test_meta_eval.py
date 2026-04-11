"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.meta_evaluation import LabelledSample, MetricResults
from healthbench_agent.domain.rubric import RubricItem
from healthbench_agent.llm_eval.meta_eval import (
    AXIS_TAG_PREFIX,
    EmptyFilterError,
    FakeJudge,
    MetricLevel,
    MetricSpec,
    adversarial_accuracy,
    adversarial_prf1,
    axis_filter,
    calibration_curve,
    cohens_kappa,
    demo_labelled_set,
    get_meta_metric,
    gold_score,
    krippendorff_alpha,
    metadata_filter,
    per_criterion_metrics,
    per_dimension_confusion,
    register_meta_metric,
    registered_meta_metrics,
    run_meta_eval,
    specialty_filter,
)


def test_metric_level_enum_values():
    assert MetricLevel.SAMPLE.value == "sample"
    assert MetricLevel.RUBRIC.value == "rubric"
    assert MetricLevel.ANY.value == "any"


def test_register_meta_metric_round_trip():
    @register_meta_metric("dummy_metric", level=MetricLevel.ANY, description="dummy")
    def _fn(df):
        return 0.0

    spec = get_meta_metric("dummy_metric")
    assert isinstance(spec, MetricSpec)
    assert spec.name == "dummy_metric"
    assert spec.level is MetricLevel.ANY
    assert spec.description == "dummy"
    assert spec.fn is _fn
    # Cleanup
    del registered_meta_metrics()["dummy_metric"]


def test_register_meta_metric_requires_level_and_description():
    with pytest.raises(TypeError):

        @register_meta_metric("missing_args")  # type: ignore[call-arg]
        def _bad(df):
            return 0.0


def test_get_meta_metric_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_meta_metric("totally_unknown_metric")


def test_axis_tag_prefix_includes_trailing_space():
    assert AXIS_TAG_PREFIX == "axis: "


def test_axis_filter_matches_tag():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: accuracy"]))
    assert not f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_axis_filter_matches_category_field():
    f = axis_filter("accuracy")
    assert f(RubricItem(criterion="x", points=1.0, category="accuracy"))


def test_axis_filter_accepts_multiple_axes():
    f = axis_filter("accuracy", "emergency")
    assert f(RubricItem(criterion="x", points=1.0, tags=["axis: emergency"]))


def test_metadata_filter_top_level_fields():
    f = metadata_filter(language="en", specialty="cardiology")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], language="en", specialty="cardiology"
    )
    assert f(sample)
    other = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], language="fr", specialty="cardiology"
    )
    assert not f(other)


def test_metadata_filter_metadata_dict_keys():
    f = metadata_filter(clinical_urgency="emergency")
    sample = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "emergency"}
    )
    assert f(sample)
    other = LabelledSample(
        prompt_id="p", prompt=[], rubrics=[], metadata={"clinical_urgency": "routine"}
    )
    assert not f(other)


def test_specialty_filter_matches_any():
    f = specialty_filter("cardiology", "pediatrics")
    assert f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="cardiology"))
    assert not f(LabelledSample(prompt_id="p", prompt=[], rubrics=[], specialty="general"))


def test_empty_filter_error_carries_filter_reprs():
    err = EmptyFilterError(sample_filter="metadata_filter(language='fr')", rubric_filter=None)
    assert "language='fr'" in str(err)


def _sample_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_gold_score_perfect_judge_returns_one():
    df = _sample_rows(
        [
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c1",
                "points": 1.0,
                "observed_met": True,
            },
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c2",
                "points": 2.0,
                "observed_met": True,
            },
        ]
    )
    assert gold_score(df) == pytest.approx(1.0)


def test_gold_score_clips_per_conversation_to_zero_when_negative():
    df = _sample_rows(
        [
            # raw = -3 / 1 = -3 → clipped to 0
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c1",
                "points": 1.0,
                "observed_met": False,
            },
            {
                "prompt_id": "p1",
                "sample_k": 1,
                "criterion": "c2",
                "points": -3.0,
                "observed_met": True,
            },
            # second conversation: perfect, raw = 1
            {
                "prompt_id": "p2",
                "sample_k": 1,
                "criterion": "c3",
                "points": 1.0,
                "observed_met": True,
            },
        ]
    )
    # mean(0.0, 1.0) == 0.5
    assert gold_score(df) == pytest.approx(0.5)


def test_gold_score_empty_dataframe_returns_zero():
    df = pd.DataFrame(columns=["prompt_id", "sample_k", "criterion", "points", "observed_met"])
    assert gold_score(df) == 0.0


def test_gold_score_registered_with_sample_level():
    spec = get_meta_metric("gold_score")
    assert spec.level is MetricLevel.SAMPLE


def _agreement_rows(observed: list[bool], expected: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(len(observed))],
            "rubric_key": [f"r{i}" for i in range(len(observed))],
            "gold_source": ["ideal_completion"] * len(observed),
            "sample_k": [1] * len(observed),
            "observed_met": observed,
            "expected_met": expected,
        }
    )


def test_cohens_kappa_full_agreement():
    df = _agreement_rows([True, False, True, False], [True, False, True, False])
    assert cohens_kappa(df) == pytest.approx(1.0)


def test_cohens_kappa_full_disagreement():
    df = _agreement_rows([True, False, True, False], [False, True, False, True])
    assert cohens_kappa(df) == pytest.approx(-1.0)


def test_cohens_kappa_random_is_near_zero():
    df = _agreement_rows([True, True, False, False], [True, False, True, False])
    assert abs(cohens_kappa(df)) < 0.5


def test_krippendorff_alpha_full_agreement():
    df = _agreement_rows([True, False, True, False], [True, False, True, False])
    assert krippendorff_alpha(df) == pytest.approx(1.0)


def test_krippendorff_alpha_random_is_near_zero():
    df = _agreement_rows([True, True, False, False], [True, False, True, False])
    assert abs(krippendorff_alpha(df)) < 0.5


def test_calibration_curve_returns_dict_keyed_by_k():
    rng = list(range(1, 8))
    rows = []
    for prompt in range(10):
        for k in rng:
            rows.append(
                {
                    "prompt_id": f"p{prompt}",
                    "rubric_key": "r1",
                    "gold_source": "ideal_completion",
                    "sample_k": k,
                    "observed_met": (prompt + k) % 2 == 0,
                    "expected_met": prompt % 2 == 0,
                }
            )
    df = pd.DataFrame(rows)
    curve = calibration_curve(df)
    assert set(curve.keys()) == {1, 3, 5, 7}
    for v in curve.values():
        assert isinstance(v, float)
        assert v >= 0


def test_calibration_curve_empty_dataframe_returns_empty_dict():
    df = pd.DataFrame(
        columns=[
            "prompt_id",
            "rubric_key",
            "gold_source",
            "sample_k",
            "observed_met",
            "expected_met",
        ]
    )
    assert calibration_curve(df) == {}


def test_calibration_curve_single_group_bucket_skipped():
    df = pd.DataFrame(
        [
            {
                "prompt_id": "p0",
                "rubric_key": "r0",
                "gold_source": "ideal_completion",
                "sample_k": 1,
                "observed_met": True,
                "expected_met": True,
            }
        ]
    )
    curve = calibration_curve(df)
    assert 1 not in curve


def test_calibration_curve_closed_form_se_for_four_groups():
    # Four groups, three agree and one disagrees, all at k=1.
    # agreements = [1, 1, 1, 0], mean = 0.75, sample variance = 0.25/3 * 4 / 3
    # actually: (3 * (1-0.75)^2 + (0-0.75)^2) / (4 - 1)
    #         = (3 * 0.0625 + 0.5625) / 3 = 0.75 / 3 = 0.25
    # SE = sqrt(0.25 / 4) = 0.25
    rows = [
        {
            "prompt_id": f"p{i}",
            "rubric_key": "r0",
            "gold_source": "ideal_completion",
            "sample_k": 1,
            "observed_met": True,
            "expected_met": (i != 3),
        }
        for i in range(4)
    ]
    df = pd.DataFrame(rows)
    curve = calibration_curve(df)
    assert curve[1] == pytest.approx(0.25, abs=1e-9)


def test_per_dimension_confusion_two_dimensions():
    df = pd.DataFrame(
        [
            {"dimension": "accuracy", "observed_met": True, "expected_met": True},
            {"dimension": "accuracy", "observed_met": True, "expected_met": False},
            {"dimension": "emergency", "observed_met": False, "expected_met": False},
            {"dimension": "emergency", "observed_met": False, "expected_met": True},
        ]
    )
    result = per_dimension_confusion(df)
    assert result["accuracy"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 0}
    assert result["emergency"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 1}


def test_per_dimension_confusion_unspecified_for_none():
    df = pd.DataFrame(
        [
            {"dimension": None, "observed_met": True, "expected_met": True},
        ]
    )
    assert per_dimension_confusion(df)["unspecified"]["tp"] == 1


def test_adversarial_accuracy_three_of_four_match():
    df = pd.DataFrame(
        [
            {"observed_met": True, "expected_met": True},
            {"observed_met": True, "expected_met": False},
            {"observed_met": False, "expected_met": False},
            {"observed_met": True, "expected_met": True},
        ]
    )
    assert adversarial_accuracy(df) == 0.75


def test_adversarial_accuracy_empty_dataframe_returns_zero():
    df = pd.DataFrame(columns=["observed_met", "expected_met"])
    assert adversarial_accuracy(df) == 0.0


def test_adversarial_prf1_returns_all_keys():
    df = pd.DataFrame(
        [
            {"observed_met": True, "expected_met": True},
            {"observed_met": False, "expected_met": True},
            {"observed_met": True, "expected_met": False},
            {"observed_met": False, "expected_met": False},
        ]
    )
    out = adversarial_prf1(df)
    assert set(out.keys()) == {"precision", "recall", "f1", "support"}
    assert out["precision"] == pytest.approx(0.5, abs=1e-9)
    assert out["recall"] == pytest.approx(0.5, abs=1e-9)
    assert out["f1"] == pytest.approx(0.5, abs=1e-9)
    assert out["support"] == pytest.approx(4.0, abs=1e-9)


def test_cohens_kappa_empty_dataframe_returns_zero():
    df = pd.DataFrame(
        columns=["prompt_id", "rubric_key", "gold_source", "observed_met", "expected_met"]
    )
    assert cohens_kappa(df) == 0.0


def test_krippendorff_alpha_empty_dataframe_returns_zero():
    df = pd.DataFrame(
        columns=["prompt_id", "rubric_key", "gold_source", "observed_met", "expected_met"]
    )
    assert krippendorff_alpha(df) == 0.0


def test_krippendorff_alpha_degenerate_all_agree_same_label():
    """When all coders agree and the label is constant, de=0 but do=0 → α=1."""
    df = _agreement_rows([True, True, True], [True, True, True])
    assert krippendorff_alpha(df) == pytest.approx(1.0)


def test_per_dimension_confusion_empty_dataframe_returns_empty_dict():
    df = pd.DataFrame(columns=["dimension", "observed_met", "expected_met"])
    assert per_dimension_confusion(df) == {}


def test_adversarial_prf1_empty_dataframe_returns_zero_fields():
    df = pd.DataFrame(columns=["observed_met", "expected_met"])
    out = adversarial_prf1(df)
    assert out == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}


def test_per_criterion_metrics_empty_dataframe_returns_empty_dict():
    df = pd.DataFrame(columns=["rubric_key", "observed_met", "expected_met"])
    assert per_criterion_metrics(df) == {}


def test_gold_score_skips_groups_with_zero_max_points():
    """When calculate_score returns None (zero positive points), the group is skipped."""
    # Two groups: one has only a penalty rubric (max_points=0 → calculate_score=None),
    # the other has a single met positive rubric → score=1.0.
    rows = [
        {
            "prompt_id": "p_zero",
            "sample_k": 1,
            "criterion": "penalty_only",
            "points": -1.0,
            "observed_met": False,
        },
        {
            "prompt_id": "p_valid",
            "sample_k": 1,
            "criterion": "good",
            "points": 1.0,
            "observed_met": True,
        },
    ]
    df = pd.DataFrame(rows)
    assert gold_score(df) == pytest.approx(1.0, abs=1e-9)


def test_calibration_curve_empty_sub_bucket_for_a_given_k():
    """If no rows have sample_k ≤ k, that k is skipped in the output curve."""
    df = pd.DataFrame(
        [
            {
                "prompt_id": "p0",
                "rubric_key": "r0",
                "gold_source": "ideal_completion",
                "sample_k": 5,
                "observed_met": True,
                "expected_met": True,
            },
            {
                "prompt_id": "p1",
                "rubric_key": "r0",
                "gold_source": "ideal_completion",
                "sample_k": 5,
                "observed_met": False,
                "expected_met": True,
            },
        ]
    )
    curve = calibration_curve(df)
    # k=1, k=3 subsets are empty and must be absent entirely.
    assert 1 not in curve
    assert 3 not in curve
    # k=5 has two observations → SE is computed.
    assert 5 in curve


def test_per_criterion_metrics_grouped_by_rubric_key():
    df = pd.DataFrame(
        [
            {"rubric_key": "c1", "observed_met": True, "expected_met": True},
            {"rubric_key": "c1", "observed_met": False, "expected_met": False},
            {"rubric_key": "c2", "observed_met": True, "expected_met": False},
        ]
    )
    out = per_criterion_metrics(df)
    assert set(out.keys()) == {"c1", "c2"}
    assert set(out["c1"].keys()) == {"accuracy", "precision", "recall", "f1"}
    assert out["c1"]["accuracy"] == pytest.approx(1.0, abs=1e-9)
    assert out["c2"]["accuracy"] == pytest.approx(0.0, abs=1e-9)


def test_fake_judge_always_met_returns_true_for_all():
    judge = FakeJudge("always_met")
    items = [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)]
    out = judge.grade([], items)
    assert all(isinstance(v, CriterionVerdict) for v in out)
    assert all(v.criteria_met for v in out)


def test_fake_judge_always_fail_returns_false_for_all():
    judge = FakeJudge("always_fail")
    items = [RubricItem(criterion="a", points=1.0)]
    out = judge.grade([], items)
    assert not out[0].criteria_met


def test_fake_judge_alternating_flips_per_call():
    judge = FakeJudge("alternating")
    out = judge.grade(
        [], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)]
    )
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_fake_judge_dict_strategy_honours_mapping():
    judge = FakeJudge({"a": True, "b": False})
    out = judge.grade(
        [], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=1.0)]
    )
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_fake_judge_callable_strategy():
    judge = FakeJudge(lambda item: item.points > 0)
    out = judge.grade(
        [], [RubricItem(criterion="a", points=1.0), RubricItem(criterion="b", points=-1.0)]
    )
    assert out[0].criteria_met is True
    assert out[1].criteria_met is False


def test_demo_labelled_set_returns_three_samples_with_mixed_flows():
    samples = demo_labelled_set()
    assert len(samples) == 3
    has_gold = sum(1 for s in samples if s.gold_response is not None)
    has_adversarial = sum(
        1 for s in samples for r in s.rubrics if r.example_meets or r.example_fails
    )
    assert has_gold >= 1
    assert has_adversarial >= 1


def _axis_extractor(item: RubricItem) -> str | None:
    return item.category


def test_run_meta_eval_with_fake_judge_produces_scores(tmp_path):
    samples = demo_labelled_set()
    perfect = {c: m for s in samples for c, m in s.expected.items()}
    view = run_meta_eval(
        FakeJudge(perfect),
        samples,
        dimension_extractor=_axis_extractor,
        n_samples=2,
        progress=False,
    )
    assert view.results.scores  # at least one metric computed
    assert view.results.n_samples_graded > 0
    assert view.results.n_rubrics_graded > 0


def test_run_meta_eval_persists_artifacts(tmp_path):
    samples = demo_labelled_set()
    view = run_meta_eval(
        FakeJudge("always_met"),
        samples,
        dimension_extractor=_axis_extractor,
        n_samples=1,
        output_dir=tmp_path,
        progress=False,
    )
    assert (tmp_path / "verdicts.parquet").exists()
    assert (tmp_path / "metrics.json").exists()
    assert view.results.verdicts_path == tmp_path / "verdicts.parquet"


def test_run_meta_eval_sample_filter_rejects_all_raises_empty_filter(tmp_path):
    samples = demo_labelled_set()
    with pytest.raises(EmptyFilterError):
        run_meta_eval(
            FakeJudge("always_met"),
            samples,
            dimension_extractor=_axis_extractor,
            sample_filter=lambda s: False,
            n_samples=1,
            progress=False,
        )


def test_run_meta_eval_rubric_filter_rejects_all_raises_empty_filter():
    samples = demo_labelled_set()
    with pytest.raises(EmptyFilterError):
        run_meta_eval(
            FakeJudge("always_met"),
            samples,
            dimension_extractor=_axis_extractor,
            rubric_filter=lambda r: False,
            n_samples=1,
            progress=False,
        )


def test_run_meta_eval_skips_sample_metrics_when_only_adversarial(caplog):
    """Sample-level metrics are dropped (with INFO log) on adversarial-only data."""
    sample = LabelledSample(
        prompt_id="adv_only",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[
            RubricItem(
                criterion="x",
                points=1.0,
                example_meets="positive",
                example_fails="negative",
                category="accuracy",
            )
        ],
    )
    view = run_meta_eval(
        FakeJudge("always_met"),
        [sample],
        dimension_extractor=_axis_extractor,
        metric_names=["gold_score", "adversarial_accuracy"],
        n_samples=1,
        progress=False,
    )
    assert "gold_score" not in view.results.scores
    assert "adversarial_accuracy" in view.results.scores


def test_run_meta_eval_raises_when_every_metric_skipped():
    sample = LabelledSample(
        prompt_id="adv_only",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[RubricItem(criterion="x", points=1.0, example_meets="ok", example_fails="bad")],
    )
    with pytest.raises(EmptyFilterError, match="gold_score"):
        run_meta_eval(
            FakeJudge("always_met"),
            [sample],
            dimension_extractor=_axis_extractor,
            metric_names=["gold_score"],
            n_samples=1,
            progress=False,
        )


def test_meta_evaluate_with_fake_judge_returns_view(monkeypatch, tmp_path):
    """meta_evaluate should accept a JudgeConfig and a pre-built dataset path."""
    from healthbench_agent.llm_eval import meta_evaluate

    # We patch the judge factory + dataset loader so the test stays offline.
    samples = demo_labelled_set()

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: samples,
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._build_judge_for_meta_eval",
        lambda config, temperature: (FakeJudge("always_met"), "fake/model@1.0", "sha"),
    )

    view = meta_evaluate(
        judge_config="config/judges/fake.yaml",
        sample_size=3,
        n_samples=1,
        cache=False,
        progress=False,
        output_dir=tmp_path,
    )
    assert view.results.scores


def test_load_subset_for_meta_eval_populates_gold_and_skips_missing(monkeypatch):
    """_load_subset_for_meta_eval should backfill gold/expected and drop rows without gold."""
    from healthbench_agent.domain.dataset import HealthBenchDataset, HealthBenchSample
    from healthbench_agent.llm_eval.meta_eval.api import _load_subset_for_meta_eval

    sample_with_gold = HealthBenchSample(
        prompt_id="with-gold",
        prompt=[{"role": "user", "content": "q1"}],
        rubrics=[
            RubricItem(criterion="good", points=2.0, tags=()),
            RubricItem(criterion="bad", points=-1.0, tags=()),
            RubricItem(criterion="zero", points=0.0, tags=()),
        ],
        example_tags=["theme:demo"],
        ideal_completions_data={"ideal_completion": "answer"},
    )
    sample_without_gold = HealthBenchSample(
        prompt_id="no-gold",
        prompt=[{"role": "user", "content": "q2"}],
        rubrics=[RubricItem(criterion="x", points=1.0, tags=())],
        example_tags=["theme:demo"],
        ideal_completions_data=None,
    )
    dataset = HealthBenchDataset(
        subset="consensus",
        samples=[sample_with_gold, sample_without_gold],
        source_path="fake.jsonl",
    )
    monkeypatch.setattr(
        "healthbench_agent.dataset.loader.load_dataset",
        lambda subset: dataset,
    )
    monkeypatch.setattr(
        "healthbench_agent.dataset.split_utils.stratified_sample",
        lambda ds, n, tag_prefix, seed: ds,
    )
    gold_lookup = {"with-gold": "gold answer", "no-gold": None}
    monkeypatch.setattr(
        "healthbench_agent.dataset.extraction.extract_ideal_completion_text",
        lambda data: gold_lookup["with-gold"] if data else None,
    )

    out = _load_subset_for_meta_eval("consensus", sample_size=2, seed=0)

    assert len(out) == 1
    assert out[0].prompt_id == "with-gold"
    assert out[0].gold_response == "gold answer"
    # Only rubrics with non-zero points become expected labels.
    assert out[0].expected == {"good": True, "bad": False}


def test_build_judge_for_meta_eval_from_path(monkeypatch):
    """_build_judge_for_meta_eval should load a JudgeConfig from a YAML path."""
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.meta_eval.api import _build_judge_for_meta_eval

    cfg = JudgeConfig(
        provider="openai",
        model="gpt-4.1",
        temperature=0.0,
        prompt_path="prompts/llm_grader/v1_llm_grader.yaml",
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.config.JudgeConfig.from_yaml",
        classmethod(lambda cls, path: cfg),
    )
    fake_judge = FakeJudge("always_met")
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge.create_judge",
        lambda c: fake_judge,
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge.load_grader_prompt",
        lambda path: ("template", {}, "prompt-sha-123"),
    )

    judge, fingerprint, prompt_sha = _build_judge_for_meta_eval(
        "config/judges/fake.yaml", temperature=0.25
    )

    assert judge is fake_judge
    assert fingerprint == "openai/gpt-4.1@0.25"
    assert prompt_sha == "prompt-sha-123"


def test_build_judge_for_meta_eval_from_config_instance(monkeypatch):
    """_build_judge_for_meta_eval should accept a pre-built JudgeConfig directly."""
    from healthbench_agent.llm_eval.grading.config import JudgeConfig
    from healthbench_agent.llm_eval.meta_eval.api import _build_judge_for_meta_eval

    cfg = JudgeConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.0,
        prompt_path="prompts/llm_grader/v1_llm_grader.yaml",
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge.create_judge",
        lambda c: FakeJudge("always_met"),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.grading.judge.load_grader_prompt",
        lambda path: ("template", {}, "sha"),
    )

    _, fingerprint, _ = _build_judge_for_meta_eval(cfg, temperature=0.5)

    assert fingerprint == "gemini/gemini-2.5-flash@0.5"


def test_build_judge_for_meta_eval_rejects_unknown_type():
    """Passing an unsupported type should raise TypeError."""
    from healthbench_agent.llm_eval.meta_eval.api import _build_judge_for_meta_eval

    with pytest.raises(TypeError, match="judge_config must be a path or JudgeConfig"):
        _build_judge_for_meta_eval(123, temperature=0.0)  # type: ignore[arg-type]


def test_meta_evaluate_with_cache_true_creates_default_verdict_cache(monkeypatch, tmp_path):
    """cache=True should construct a default VerdictCache and pass it to run_meta_eval."""
    from healthbench_agent.llm_eval import meta_evaluate
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: demo_labelled_set(),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._build_judge_for_meta_eval",
        lambda config, temperature: (FakeJudge("always_met"), "fake/model@1.0", "sha"),
    )
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return MetricResults(
            scores={"gold_score": 0.5},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "fake/model@1.0"},
        )

    monkeypatch.setattr("healthbench_agent.llm_eval.meta_eval.api.run_meta_eval", fake_run)

    meta_evaluate(
        judge_config="config/judges/fake.yaml",
        sample_size=1,
        n_samples=1,
        cache=True,
        output_dir=tmp_path,
    )
    assert isinstance(captured["cache"], VerdictCache)
    assert captured["cache"].enabled is True


def test_meta_evaluate_with_explicit_cache_instance_passes_through(monkeypatch, tmp_path):
    """A pre-built VerdictCache should be forwarded unchanged."""
    from healthbench_agent.llm_eval import meta_evaluate
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    explicit = VerdictCache(root=tmp_path / "custom", enabled=True)

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: demo_labelled_set(),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.api._build_judge_for_meta_eval",
        lambda config, temperature: (FakeJudge("always_met"), "fake/model@1.0", "sha"),
    )
    captured: dict[str, Any] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return MetricResults(
            scores={"gold_score": 0.5},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "fake/model@1.0"},
        )

    monkeypatch.setattr("healthbench_agent.llm_eval.meta_eval.api.run_meta_eval", fake_run)

    meta_evaluate(
        judge_config="config/judges/fake.yaml",
        sample_size=1,
        n_samples=1,
        cache=explicit,
        output_dir=tmp_path,
    )
    assert captured["cache"] is explicit


def test_cli_run_dispatches_to_run_meta_eval(monkeypatch, tmp_path, capsys):
    """CLI run subcommand should call run_meta_eval with parsed args."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    called: dict[str, Any] = {}

    def fake_run(**kwargs):
        called.update(kwargs)
        results = MetricResults(
            scores={"gold_score": 0.5},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "fake"},
        )
        from healthbench_agent.llm_eval.meta_eval.results_view import MetricResultsView

        return MetricResultsView(results=results)

    monkeypatch.setattr("healthbench_agent.llm_eval.cli.meta_eval.run_meta_eval", fake_run)
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_judge_for_cli",
        lambda config, temperature: (FakeJudge("always_met"), "fake@1", "sha"),
    )

    cli_meta_eval.main(
        [
            "run",
            "--judge-config",
            "fake.yaml",
            "--sample-size",
            "1",
            "--n-samples",
            "1",
            "--no-cache",
            "--no-mlflow",
            "--no-progress",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert called["n_samples"] == 1
    captured = capsys.readouterr()
    assert "gold_score" in captured.out


def test_cli_list_metrics_prints_all(capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    cli_meta_eval.main(["list-metrics"])
    out = capsys.readouterr().out
    for name in (
        "gold_score",
        "cohens_kappa",
        "krippendorff_alpha",
        "calibration_curve",
        "per_dimension_confusion",
        "adversarial_accuracy",
        "adversarial_prf1",
        "per_criterion_metrics",
    ):
        assert name in out


def test_cli_clear_cache_removes_directory(tmp_path, monkeypatch):
    from healthbench_agent.llm_eval.cache import VerdictCache
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    cache = VerdictCache(root=tmp_path / "cache", enabled=True)
    cache.put(
        cache.make_key("m", "s", [{"role": "user", "content": "x"}], "rt", 1),
        CriterionVerdict(criterion="rt", criteria_met=True, explanation=""),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._default_cache_for_cli",
        lambda: cache,
    )
    cli_meta_eval.main(["clear-cache"])
    assert not cache.root.exists() or not any(cache.root.iterdir())


def test_cli_regenerate_replays_metrics(tmp_path):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    from healthbench_agent.llm_eval.meta_eval.results_io import load_results

    samples = demo_labelled_set()
    view = run_meta_eval(
        FakeJudge({c: m for s in samples for c, m in s.expected.items()}),
        samples,
        dimension_extractor=lambda r: r.category,
        n_samples=1,
        output_dir=tmp_path,
        progress=False,
    )
    # Mutate the metrics.json so we can prove regenerate overwrote it.
    import json as _json

    (tmp_path / "metrics.json").write_text(_json.dumps({**view.results.to_dict(), "scores": {}}))
    cli_meta_eval.main(["regenerate", str(tmp_path)])
    reloaded = load_results(tmp_path)
    assert reloaded.results.scores  # repopulated


def test_cli_compare_prints_diff_table(tmp_path, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    from healthbench_agent.llm_eval.meta_eval.results_io import save_results
    from healthbench_agent.llm_eval.meta_eval.results_view import MetricResultsView

    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_view = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.8},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "openai"},
        )
    )
    b_view = MetricResultsView(
        results=MetricResults(
            scores={"gold_score": 0.7},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "google"},
        )
    )
    save_results(a_view.results, a_dir)
    save_results(b_view.results, b_dir)
    cli_meta_eval.main(["compare", str(a_dir), str(b_dir)])
    out = capsys.readouterr().out
    assert "gold_score" in out
    assert "delta" in out or "0.8" in out


def test_cli_list_metadata_keys(monkeypatch, capsys):
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    cli_meta_eval.main(["list-metadata-keys", "--sample-size", "3"])
    out = capsys.readouterr().out
    # Structural headers must be present.
    assert "Top-level fields:" in out
    assert "metadata dict keys:" in out
    # demo_labelled_set() populates language='en', specialty, and
    # metadata={'clinical_urgency': ...} — all three should surface.
    assert "'en'" in out
    assert "clinical_urgency" in out


def test_cli_dry_run_does_not_call_judge(monkeypatch, capsys):
    from healthbench_agent.domain.judge import JudgeGrader
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    judge_calls = {"n": 0}

    class CountingJudge(JudgeGrader):
        def grade(self, conversation, rubric_items):
            judge_calls["n"] += 1
            return [
                CriterionVerdict(criterion=r.criterion, criteria_met=True, explanation="")
                for r in rubric_items
            ]

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), lambda r: r.category),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_judge_for_cli",
        lambda config, temperature: (CountingJudge(), "fake@1", "sha"),
    )

    cli_meta_eval.main(
        [
            "run",
            "--judge-config",
            "fake.yaml",
            "--sample-size",
            "3",
            "--n-samples",
            "1",
            "--no-cache",
            "--no-progress",
            "--no-mlflow",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "DRY RUN" in out.upper() or "Dry run" in out
    assert judge_calls["n"] == 0


def test_cli_default_subcommand_inferred_from_leading_flag(monkeypatch, capsys):
    """Bare `meta-evaluate-judge --judge-config foo.yaml` injects 'run'."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._cmd_run",
        lambda args: print("RUN_CALLED"),
    )
    cli_meta_eval.main(["--judge-config", "x.yaml", "--sample-size", "1"])
    assert "RUN_CALLED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# runner.py coverage backfill
# ---------------------------------------------------------------------------


def test_run_meta_eval_rubric_filter_keeps_subset_of_rubrics():
    """A rubric_filter that keeps some rubrics exercises the append branch."""
    sample = LabelledSample(
        prompt_id="mixed",
        prompt=[{"role": "user", "content": "q"}],
        rubrics=[
            RubricItem(criterion="keep", points=1.0, category="accuracy", tags=()),
            RubricItem(criterion="drop", points=1.0, category="emergency", tags=()),
        ],
        gold_response="gold",
        expected={"keep": True, "drop": True},
    )
    view = run_meta_eval(
        FakeJudge("always_met"),
        [sample],
        dimension_extractor=_axis_extractor,
        rubric_filter=lambda r: r.criterion == "keep",
        metric_names=["gold_score"],
        n_samples=1,
        progress=False,
    )
    # The surviving sample retained exactly one rubric.
    assert view.results.n_samples_graded == 1
    assert view.results.n_rubrics_graded == 1


def test_run_meta_eval_with_cache_requires_fingerprint_and_prompt_sha(tmp_path):
    """Passing a cache without both fingerprint and prompt sha must raise."""
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    cache = VerdictCache(root=tmp_path / "cache", enabled=True)
    with pytest.raises(ValueError, match="model_fingerprint and judge_prompt_sha"):
        run_meta_eval(
            FakeJudge("always_met"),
            demo_labelled_set(),
            dimension_extractor=_axis_extractor,
            n_samples=1,
            cache=cache,
            progress=False,
        )


def test_run_meta_eval_with_cache_wraps_judge_in_cached_grader(tmp_path):
    """With cache + fingerprint + prompt_sha, grade_for_k wraps in CachedJudgeGrader."""
    from healthbench_agent.llm_eval.cache.store import VerdictCache

    cache = VerdictCache(root=tmp_path / "cache", enabled=True)
    view = run_meta_eval(
        FakeJudge("always_met"),
        demo_labelled_set(),
        dimension_extractor=_axis_extractor,
        n_samples=1,
        cache=cache,
        model_fingerprint="fake/model@1.0",
        judge_prompt_sha="sha-abc",
        progress=False,
    )
    # Cache stats flow through to judge_metadata.
    assert "cache_hits" in view.results.judge_metadata
    assert "cache_misses" in view.results.judge_metadata
    # Running the same invocation twice should see cache hits > 0.
    view2 = run_meta_eval(
        FakeJudge("always_met"),
        demo_labelled_set(),
        dimension_extractor=_axis_extractor,
        n_samples=1,
        cache=cache,
        model_fingerprint="fake/model@1.0",
        judge_prompt_sha="sha-abc",
        progress=False,
    )
    assert view2.results.judge_metadata["cache_hits"] > 0


def test_run_meta_eval_with_progress_true_uses_thread_map(monkeypatch):
    """progress=True should dispatch via tqdm's thread_map helper."""
    called: dict[str, Any] = {}

    def fake_thread_map(fn, pairs, **kwargs):
        called["invoked"] = True
        called["max_workers"] = kwargs.get("max_workers")
        return [fn(pair) for pair in pairs]

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval.runner.thread_map",
        fake_thread_map,
    )
    run_meta_eval(
        FakeJudge("always_met"),
        demo_labelled_set(),
        dimension_extractor=_axis_extractor,
        n_samples=1,
        progress=True,
        meta_eval_max_workers=4,
    )
    assert called["invoked"] is True
    assert called["max_workers"] == 4


# ---------------------------------------------------------------------------
# cli/meta_eval.py coverage backfill
# ---------------------------------------------------------------------------


def test_load_consensus_labelled_returns_samples_and_axis_extractor(monkeypatch):
    """_load_consensus_labelled delegates and builds an axis extractor."""
    from healthbench_agent.llm_eval.cli.meta_eval import _load_consensus_labelled

    canned = demo_labelled_set()
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: canned,
    )
    samples, extractor = _load_consensus_labelled("consensus", sample_size=3, seed=0)
    assert samples is canned
    # The extractor reads axis tags when present.
    tagged = RubricItem(criterion="x", points=1.0, tags=("axis: accuracy",))
    assert extractor(tagged) == "accuracy"
    # Falls back to category when no axis tag is present.
    categorised = RubricItem(criterion="x", points=1.0, category="emergency", tags=())
    assert extractor(categorised) == "emergency"


def test_load_consensus_labelled_exits_when_all_samples_dropped(monkeypatch, capsys):
    """Empty dataset after gold extraction should exit with code 2."""
    from healthbench_agent.llm_eval.cli.meta_eval import _load_consensus_labelled

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval._load_subset_for_meta_eval",
        lambda subset, sample_size, seed: [],
    )
    with pytest.raises(SystemExit) as excinfo:
        _load_consensus_labelled("consensus", sample_size=1, seed=0)
    assert excinfo.value.code == 2
    assert "physician ideal completions" in capsys.readouterr().err


def test_build_judge_for_cli_delegates_to_meta_eval_helper(monkeypatch):
    """_build_judge_for_cli forwards to _build_judge_for_meta_eval."""
    from healthbench_agent.llm_eval.cli.meta_eval import _build_judge_for_cli

    captured: dict[str, Any] = {}

    def fake_builder(config, temperature):
        captured["config"] = config
        captured["temperature"] = temperature
        return (FakeJudge("always_met"), "fake@1", "prompt-sha")

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.meta_eval._build_judge_for_meta_eval",
        fake_builder,
    )
    judge, fingerprint, prompt_sha = _build_judge_for_cli("fake.yaml", 0.25)
    assert captured == {"config": "fake.yaml", "temperature": 0.25}
    assert fingerprint == "fake@1"
    assert prompt_sha == "prompt-sha"
    assert isinstance(judge, FakeJudge)


def test_build_filters_parses_metadata_key_value_pairs():
    """KEY=VALUE entries should become a metadata_filter closure."""
    import argparse

    from healthbench_agent.llm_eval.cli.meta_eval import _build_filters

    args = argparse.Namespace(rubric_axis=["accuracy"], metadata=["language=en"])
    sf, rf = _build_filters(args)
    assert sf is not None
    assert rf is not None
    # The sample filter should accept a sample whose language matches.
    sample = LabelledSample(prompt_id="p", prompt=[], rubrics=[], language="en")
    assert sf(sample)
    other = LabelledSample(prompt_id="p", prompt=[], rubrics=[], language="fr")
    assert not sf(other)


def test_build_filters_rejects_malformed_metadata_entries():
    """--metadata without '=' should raise SystemExit with a helpful message."""
    import argparse

    from healthbench_agent.llm_eval.cli.meta_eval import _build_filters

    args = argparse.Namespace(rubric_axis=[], metadata=["malformed"])
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _build_filters(args)


def test_default_cache_for_cli_returns_enabled_verdict_cache():
    """_default_cache_for_cli yields an enabled VerdictCache pointing at the default root."""
    from healthbench_agent.llm_eval.cache.store import VerdictCache
    from healthbench_agent.llm_eval.cli.meta_eval import _default_cache_for_cli

    cache = _default_cache_for_cli()
    assert isinstance(cache, VerdictCache)
    assert cache.enabled is True


def test_cli_run_surfaces_empty_filter_error_as_system_exit(monkeypatch, tmp_path, capsys):
    """_cmd_run must translate EmptyFilterError into SystemExit(2) with stderr output."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    def raise_empty(**kwargs):
        raise EmptyFilterError(sample_filter="sf", rubric_filter="rf")

    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: (demo_labelled_set(), _axis_extractor),
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._build_judge_for_cli",
        lambda config, temperature: (FakeJudge("always_met"), "fake@1", "sha"),
    )
    monkeypatch.setattr("healthbench_agent.llm_eval.cli.meta_eval.run_meta_eval", raise_empty)

    with pytest.raises(SystemExit) as excinfo:
        cli_meta_eval.main(
            [
                "run",
                "--judge-config",
                "fake.yaml",
                "--sample-size",
                "1",
                "--n-samples",
                "1",
                "--no-cache",
                "--no-progress",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert excinfo.value.code == 2
    assert "sf" in capsys.readouterr().err


def test_cli_regenerate_missing_parquet_exits(tmp_path, capsys):
    """regenerate must exit cleanly when verdicts.parquet is absent."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    with pytest.raises(SystemExit) as excinfo:
        cli_meta_eval.main(["regenerate", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "verdicts.parquet not found" in capsys.readouterr().err


def test_cli_regenerate_unknown_metric_exits(tmp_path, capsys):
    """regenerate must exit cleanly when --metrics names an unknown metric."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    # Seed a valid run dir first.
    samples = demo_labelled_set()
    run_meta_eval(
        FakeJudge("always_met"),
        samples,
        dimension_extractor=_axis_extractor,
        n_samples=1,
        output_dir=tmp_path,
        progress=False,
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_meta_eval.main(["regenerate", str(tmp_path), "--metrics", "totally_unknown_metric"])
    assert excinfo.value.code == 2
    assert "totally_unknown_metric" in capsys.readouterr().err


def test_cli_regenerate_skips_level_with_no_matching_rows(tmp_path):
    """regenerate should skip a metric whose level has no matching verdict rows.

    The observable effect is that the skipped metric is absent from the
    persisted metrics.json after regeneration.
    """
    import pandas as pd

    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    from healthbench_agent.llm_eval.meta_eval.results_io import load_results, save_results

    # Write a parquet file that contains only ideal_completion rows, then
    # request a rubric-level metric which will have zero matching rows.
    parquet_rows = [
        {
            "prompt_id": "p1",
            "sample_k": 1,
            "rubric_key": "c1",
            "criterion": "c1",
            "points": 1.0,
            "observed_met": True,
            "expected_met": True,
            "gold_source": "ideal_completion",
            "dimension": "accuracy",
        }
    ]
    pd.DataFrame(parquet_rows).to_parquet(tmp_path / "verdicts.parquet")
    save_results(
        MetricResults(
            scores={"gold_score": 1.0},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "fake"},
        ),
        tmp_path,
    )

    cli_meta_eval.main(["regenerate", str(tmp_path), "--metrics", "adversarial_accuracy"])

    # Observable behaviour: the skipped metric must not appear in the
    # persisted metrics.json after regenerate.
    reloaded = load_results(tmp_path)
    assert "adversarial_accuracy" not in reloaded.results.scores


def test_cli_compare_missing_run_dir_exits(tmp_path, capsys):
    """compare must exit cleanly when one of the run dirs has no metrics.json."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    bogus = tmp_path / "bogus"
    bogus.mkdir()
    # tmp_path itself has no metrics.json either.
    with pytest.raises(SystemExit) as excinfo:
        cli_meta_eval.main(["compare", str(bogus), str(tmp_path)])
    assert excinfo.value.code == 2
    assert "metrics.json" in capsys.readouterr().err


def test_cli_compare_writes_markdown_output(tmp_path, capsys):
    """compare --output should serialise the diff as markdown next to the call."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval
    from healthbench_agent.llm_eval.meta_eval.results_io import save_results

    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    output = tmp_path / "diff.md"
    save_results(
        MetricResults(
            scores={"gold_score": 0.8},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "alpha"},
        ),
        a_dir,
    )
    save_results(
        MetricResults(
            scores={"gold_score": 0.7},
            n_samples_graded=1,
            n_rubrics_graded=1,
            judge_metadata={"judge_model": "beta"},
        ),
        b_dir,
    )
    cli_meta_eval.main(["compare", str(a_dir), str(b_dir), "--output", str(output)])
    assert output.exists()
    body = output.read_text()
    assert "gold_score" in body
    assert "Markdown table written to" in capsys.readouterr().out


def test_fake_judge_rejects_unknown_strategy():
    """FakeJudge raises ValueError for an unrecognised strategy string."""
    judge = FakeJudge("definitely_not_a_strategy")
    with pytest.raises(ValueError, match="Unknown FakeJudge strategy"):
        judge.grade(
            [{"role": "user", "content": "?"}],
            [RubricItem(criterion="x", points=1.0)],
        )


def test_cli_list_metadata_keys_prints_none_when_metadata_empty(monkeypatch, capsys):
    """list-metadata-keys should print '(none)' when no metadata keys are present."""
    from healthbench_agent.llm_eval.cli import meta_eval as cli_meta_eval

    # Sample with only a prompt — no language, specialty, persona, or metadata dict.
    empty_sample = LabelledSample(
        prompt_id="empty",
        prompt=[{"role": "user", "content": "?"}],
        rubrics=[],
    )
    monkeypatch.setattr(
        "healthbench_agent.llm_eval.cli.meta_eval._load_consensus_labelled",
        lambda subset, sample_size, seed: ([empty_sample], _axis_extractor),
    )
    cli_meta_eval.main(["list-metadata-keys", "--sample-size", "1"])
    out = capsys.readouterr().out
    # All three top-level fields (language, specialty, user_persona) are
    # None, plus the metadata dict is empty → '(none)' prints exactly four
    # times: once per top-level field and once for the metadata dict.
    assert out.count("(none)") == 4
    assert "language: (none)" in out
    assert "specialty: (none)" in out
    assert "user_persona: (none)" in out
