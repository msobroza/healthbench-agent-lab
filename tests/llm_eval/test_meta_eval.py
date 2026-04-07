"""Tests for the meta_eval module: registry, metrics, runner, filters, CLI."""

from __future__ import annotations

import pandas as pd
import pytest

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.meta_evaluation import LabelledSample
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
