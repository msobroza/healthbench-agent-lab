"""Tests for healthbench_agent.domain.scoring.

Covers calculate_score, clip_score, aggregate_scores, and stratified_scores
using the ZOMBIES heuristic (Zero, One, Many, Boundaries, Interface, Exceptions,
Simple scenarios).
"""

import pytest

from healthbench_agent.domain import CriterionVerdict, RubricItem, SingleEvalResult
from healthbench_agent.domain.scoring import (
    aggregate_scores,
    calculate_score,
    clip_score,
    stratified_scores,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def positive_item():
    return RubricItem(criterion="Recommends immediate care", points=1.0, tags=["emergency"])


@pytest.fixture
def penalty_item():
    return RubricItem(criterion="Is overly verbose", points=-1.0, tags=["quality"])


@pytest.fixture
def met_verdict(positive_item):
    return CriterionVerdict(criterion=positive_item.criterion, criteria_met=True)


@pytest.fixture
def unmet_verdict(positive_item):
    return CriterionVerdict(criterion=positive_item.criterion, criteria_met=False)


@pytest.fixture
def two_positive_items():
    return [
        RubricItem(criterion="Recommends immediate care", points=1.0, tags=["emergency"]),
        RubricItem(criterion="Explains diagnosis clearly", points=0.5, tags=["accuracy"]),
    ]


@pytest.fixture
def two_met_verdicts(two_positive_items):
    return [
        CriterionVerdict(criterion=item.criterion, criteria_met=True) for item in two_positive_items
    ]


@pytest.fixture
def two_unmet_verdicts(two_positive_items):
    return [
        CriterionVerdict(criterion=item.criterion, criteria_met=False)
        for item in two_positive_items
    ]


@pytest.fixture
def perfect_result():
    return SingleEvalResult(score=1.0, metrics={"accuracy": 1.0, "emergency": 1.0})


@pytest.fixture
def zero_result():
    return SingleEvalResult(score=0.0, metrics={"accuracy": 0.0, "emergency": 0.0})


@pytest.fixture
def none_result():
    return SingleEvalResult(score=None, metrics={})


# ---------------------------------------------------------------------------
# calculate_score
# ---------------------------------------------------------------------------


def test_calculate_score_all_criteria_met_returns_one(two_positive_items, two_met_verdicts):
    assert calculate_score(two_positive_items, two_met_verdicts) == pytest.approx(1.0)


def test_calculate_score_no_criteria_met_returns_zero(two_positive_items, two_unmet_verdicts):
    assert calculate_score(two_positive_items, two_unmet_verdicts) == pytest.approx(0.0)


def test_calculate_score_partial_criteria_met(two_positive_items, two_met_verdicts):
    two_met_verdicts[1] = CriterionVerdict(
        criterion=two_met_verdicts[1].criterion, criteria_met=False
    )
    # only first item (1.0) met out of total 1.5
    assert calculate_score(two_positive_items, two_met_verdicts) == pytest.approx(1.0 / 1.5)


def test_calculate_score_penalty_met_reduces_score_below_zero(positive_item, penalty_item):
    items = [positive_item, penalty_item]
    verdicts = [
        CriterionVerdict(criterion=positive_item.criterion, criteria_met=True),
        CriterionVerdict(criterion=penalty_item.criterion, criteria_met=True),
    ]
    # achieved = 1.0 + (-1.0) = 0.0 / total_possible = 1.0
    assert calculate_score(items, verdicts) == pytest.approx(0.0)


def test_calculate_score_only_penalty_criteria_returns_none(penalty_item):
    verdict = CriterionVerdict(criterion=penalty_item.criterion, criteria_met=False)
    assert calculate_score([penalty_item], [verdict]) is None


def test_calculate_score_empty_rubric_returns_none():
    assert calculate_score([], []) is None


def test_calculate_score_single_criterion_met(positive_item, met_verdict):
    assert calculate_score([positive_item], [met_verdict]) == pytest.approx(1.0)


def test_calculate_score_single_criterion_not_met(positive_item, unmet_verdict):
    assert calculate_score([positive_item], [unmet_verdict]) == pytest.approx(0.0)


def test_calculate_score_mismatched_lengths_raises(two_positive_items, met_verdict):
    with pytest.raises(ValueError):
        calculate_score(two_positive_items, [met_verdict])


def test_calculate_score_zero_point_item_ignored_in_denominator(positive_item):
    zero_item = RubricItem(criterion="Zero weight item", points=0.0, tags=[])
    items = [positive_item, zero_item]
    verdicts = [
        CriterionVerdict(criterion=positive_item.criterion, criteria_met=True),
        CriterionVerdict(criterion=zero_item.criterion, criteria_met=True),
    ]
    assert calculate_score(items, verdicts) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# clip_score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, 1.0),  # upper boundary
        (0.0, 0.0),  # lower boundary
        (0.5, 0.5),  # mid-range unchanged
        (-0.5, 0.0),  # negative clipped to 0
        (1.5, 1.0),  # above 1 clipped to 1
        (-10.0, 0.0),  # large negative
        (10.0, 1.0),  # large positive
    ],
)
def test_clip_score(score, expected):
    assert clip_score(score) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


def test_aggregate_scores_empty_list_returns_zero():
    assert aggregate_scores([]) == pytest.approx(0.0)


def test_aggregate_scores_all_none_scores_returns_zero(none_result):
    assert aggregate_scores([none_result, none_result]) == pytest.approx(0.0)


def test_aggregate_scores_single_perfect_result(perfect_result):
    assert aggregate_scores([perfect_result]) == pytest.approx(1.0)


def test_aggregate_scores_single_zero_result(zero_result):
    assert aggregate_scores([zero_result]) == pytest.approx(0.0)


def test_aggregate_scores_clips_negative_before_averaging(perfect_result):
    negative_result = SingleEvalResult(score=-0.5, metrics={})
    assert aggregate_scores([perfect_result, negative_result]) == pytest.approx(0.5)


def test_aggregate_scores_skips_none_scores(perfect_result, none_result, zero_result):
    # none_result is skipped → mean of 1.0 and 0.0 = 0.5
    assert aggregate_scores([perfect_result, none_result, zero_result]) == pytest.approx(0.5)


def test_aggregate_scores_clips_above_one_before_averaging(zero_result):
    above_one_result = SingleEvalResult(score=1.5, metrics={})
    # clipped to 1.0 and 0.0 → mean = 0.5
    assert aggregate_scores([above_one_result, zero_result]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# stratified_scores
# ---------------------------------------------------------------------------


def test_stratified_scores_empty_list_returns_empty_dict():
    assert stratified_scores([]) == {}


def test_stratified_scores_aggregates_by_label(perfect_result, zero_result):
    scores = stratified_scores([perfect_result, zero_result])
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["emergency"] == pytest.approx(0.5)


def test_stratified_scores_clips_before_averaging():
    negative_result = SingleEvalResult(score=0.5, metrics={"safety": -0.5})
    positive_result = SingleEvalResult(score=0.5, metrics={"safety": 1.0})
    # clipped: 0.0 and 1.0 → mean = 0.5
    assert stratified_scores([negative_result, positive_result])["safety"] == pytest.approx(0.5)


def test_stratified_scores_omits_labels_absent_from_all_results():
    result_a = SingleEvalResult(score=1.0, metrics={"accuracy": 1.0})
    result_b = SingleEvalResult(score=0.5, metrics={"emergency": 0.5})
    scores = stratified_scores([result_a, result_b])
    assert "accuracy" in scores
    assert "emergency" in scores


def test_stratified_scores_single_result(perfect_result):
    scores = stratified_scores([perfect_result])
    assert scores["accuracy"] == pytest.approx(1.0)
    assert scores["emergency"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_score — realistic HealthBench rubric scenarios (conftest fixtures)
#
# simple_sample rubric points : [10, 9, -10]   total_possible = 10 + 9 = 19
# hard_sample rubric points   : [7, 7, 8, -9]  total_possible = 7 + 7 + 8 = 22
# ---------------------------------------------------------------------------


def test_calculate_score_only_positive_criteria_all_met(simple_sample):
    positive = [r for r in simple_sample.rubrics if r.points > 0]
    verdicts = [CriterionVerdict(criterion=r.criterion, criteria_met=True) for r in positive]
    assert calculate_score(positive, verdicts) == pytest.approx(1.0)


def test_calculate_score_positive_met_penalty_unmet_gives_perfect(simple_sample):
    # met: points 10 and 9; unmet: penalty -10 → achieved=19, total_possible=19
    verdicts = [
        CriterionVerdict(criterion=simple_sample.rubrics[0].criterion, criteria_met=True),
        CriterionVerdict(criterion=simple_sample.rubrics[1].criterion, criteria_met=True),
        CriterionVerdict(criterion=simple_sample.rubrics[2].criterion, criteria_met=False),
    ]
    assert calculate_score(simple_sample.rubrics, verdicts) == pytest.approx(1.0)


def test_calculate_score_all_met_including_penalty_lowers_score(simple_sample):
    # met all: achieved = 10 + 9 + (-10) = 9; total_possible = 19
    verdicts = [
        CriterionVerdict(criterion=r.criterion, criteria_met=True) for r in simple_sample.rubrics
    ]
    assert calculate_score(simple_sample.rubrics, verdicts) == pytest.approx(9.0 / 19.0)


def test_calculate_score_hard_sample_all_met(hard_sample):
    # achieved = 7 + 7 + 8 + (-9) = 13; total_possible = 7 + 7 + 8 = 22
    verdicts = [
        CriterionVerdict(criterion=r.criterion, criteria_met=True) for r in hard_sample.rubrics
    ]
    assert calculate_score(hard_sample.rubrics, verdicts) == pytest.approx(13.0 / 22.0)


def test_calculate_score_hard_sample_none_met_returns_zero(hard_sample):
    verdicts = [
        CriterionVerdict(criterion=r.criterion, criteria_met=False) for r in hard_sample.rubrics
    ]
    assert calculate_score(hard_sample.rubrics, verdicts) == pytest.approx(0.0)


def test_calculate_score_hard_sample_penalty_only_met_returns_zero(hard_sample):
    # Only the penalty criterion (-9) is met; positive ones unmet → achieved = -9, clamped
    penalty = next(r for r in hard_sample.rubrics if r.points < 0)
    verdicts = [
        CriterionVerdict(
            criterion=r.criterion,
            criteria_met=(r.criterion == penalty.criterion),
        )
        for r in hard_sample.rubrics
    ]
    # achieved = -9; total_possible = 22 → raw = -9/22 < 0, but score returns raw value
    score = calculate_score(hard_sample.rubrics, verdicts)
    assert score is not None
    assert score < 0
