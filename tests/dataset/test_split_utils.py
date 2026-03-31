"""Tests for healthbench_agent.dataset.split_utils.

Covers sample_dataset and stratified_sample using the ZOMBIES heuristic.
All tests are deterministic — seeds are fixed.
"""

from __future__ import annotations

import pytest

from healthbench_agent.dataset.split_utils import sample_dataset, stratified_sample
from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample, RubricItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(prompt_id: str, tags: list[str] | None = None) -> HealthBenchSample:
    return HealthBenchSample(
        prompt_id=prompt_id,
        prompt=[{"role": "user", "content": "question"}],
        rubrics=[RubricItem(criterion="c", points=1.0, tags=[])],
        example_tags=tags or [],
    )


@pytest.fixture
def ten_sample_dataset():
    samples = [_make_sample(f"p{i:03d}") for i in range(10)]
    return HealthBenchDataset(subset="main", samples=samples, source_path="/tmp/main.jsonl")


@pytest.fixture
def themed_dataset():
    """Dataset with 6 cardiology + 4 oncology samples tagged by theme."""
    samples = (
        [_make_sample(f"card_{i}", tags=["theme: cardiology"]) for i in range(6)]
        + [_make_sample(f"onco_{i}", tags=["theme: oncology"]) for i in range(4)]
    )
    return HealthBenchDataset(subset="main", samples=samples, source_path="/tmp/main.jsonl")


@pytest.fixture
def empty_dataset():
    return HealthBenchDataset(subset="hard", samples=[], source_path="/tmp/hard.jsonl")


# ---------------------------------------------------------------------------
# sample_dataset — Zero / boundary
# ---------------------------------------------------------------------------


def test_sample_dataset_n_zero_returns_empty(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=0)
    assert len(result) == 0


def test_sample_dataset_negative_n_raises_value_error(ten_sample_dataset):
    with pytest.raises(ValueError, match="non-negative"):
        sample_dataset(ten_sample_dataset, n=-1)


def test_sample_dataset_n_greater_than_size_returns_all(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=100)
    assert len(result) == 10


def test_sample_dataset_empty_dataset_returns_empty(empty_dataset):
    result = sample_dataset(empty_dataset, n=5)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# sample_dataset — One / Many
# ---------------------------------------------------------------------------


def test_sample_dataset_n_one_returns_single_sample(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=1)
    assert len(result) == 1


def test_sample_dataset_n_five_returns_five(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=5)
    assert len(result) == 5


def test_sample_dataset_samples_are_from_source(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=5)
    source_ids = {s.prompt_id for s in ten_sample_dataset.samples}
    for sample in result.samples:
        assert sample.prompt_id in source_ids


# ---------------------------------------------------------------------------
# sample_dataset — reproducibility and metadata
# ---------------------------------------------------------------------------


def test_sample_dataset_same_seed_returns_same_result(ten_sample_dataset):
    r1 = sample_dataset(ten_sample_dataset, n=5, seed=42)
    r2 = sample_dataset(ten_sample_dataset, n=5, seed=42)
    assert [s.prompt_id for s in r1.samples] == [s.prompt_id for s in r2.samples]


def test_sample_dataset_different_seeds_return_different_results(ten_sample_dataset):
    r1 = sample_dataset(ten_sample_dataset, n=5, seed=0)
    r2 = sample_dataset(ten_sample_dataset, n=5, seed=99)
    # Very likely to differ with 10 items choose 5
    assert [s.prompt_id for s in r1.samples] != [s.prompt_id for s in r2.samples]


def test_sample_dataset_preserves_subset_label(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=3)
    assert result.subset == "main"


def test_sample_dataset_preserves_source_path(ten_sample_dataset):
    result = sample_dataset(ten_sample_dataset, n=3)
    assert result.source_path == ten_sample_dataset.source_path


def test_sample_dataset_does_not_mutate_source(ten_sample_dataset):
    original_ids = [s.prompt_id for s in ten_sample_dataset.samples]
    sample_dataset(ten_sample_dataset, n=5)
    assert [s.prompt_id for s in ten_sample_dataset.samples] == original_ids


# ---------------------------------------------------------------------------
# stratified_sample — Zero / boundary
# ---------------------------------------------------------------------------


def test_stratified_sample_n_zero_returns_empty(themed_dataset):
    result = stratified_sample(themed_dataset, n=0, tag_prefix="theme")
    assert len(result) == 0


def test_stratified_sample_negative_n_raises_value_error(themed_dataset):
    with pytest.raises(ValueError, match="non-negative"):
        stratified_sample(themed_dataset, n=-1, tag_prefix="theme")


def test_stratified_sample_empty_dataset_returns_empty(empty_dataset):
    result = stratified_sample(empty_dataset, n=5, tag_prefix="theme")
    assert len(result) == 0


# ---------------------------------------------------------------------------
# stratified_sample — proportional coverage
# ---------------------------------------------------------------------------


def test_stratified_sample_includes_both_strata(themed_dataset):
    result = stratified_sample(themed_dataset, n=6, tag_prefix="theme")
    result_tags = [s.example_tags[0] for s in result.samples if s.example_tags]
    assert any("cardiology" in t for t in result_tags)
    assert any("oncology" in t for t in result_tags)


def test_stratified_sample_result_is_not_larger_than_n(themed_dataset):
    result = stratified_sample(themed_dataset, n=4, tag_prefix="theme")
    assert len(result) <= 4


def test_stratified_sample_unknown_prefix_groups_all_under_no_prefix(ten_sample_dataset):
    # ten_sample_dataset has no example_tags → all go to _no_prefix
    result = stratified_sample(ten_sample_dataset, n=5, tag_prefix="theme")
    assert len(result) <= 5


# ---------------------------------------------------------------------------
# stratified_sample — reproducibility and metadata
# ---------------------------------------------------------------------------


def test_stratified_sample_same_seed_returns_same_result(themed_dataset):
    r1 = stratified_sample(themed_dataset, n=6, tag_prefix="theme", seed=7)
    r2 = stratified_sample(themed_dataset, n=6, tag_prefix="theme", seed=7)
    assert [s.prompt_id for s in r1.samples] == [s.prompt_id for s in r2.samples]


def test_stratified_sample_preserves_subset_label(themed_dataset):
    result = stratified_sample(themed_dataset, n=4, tag_prefix="theme")
    assert result.subset == "main"


def test_stratified_sample_does_not_mutate_source(themed_dataset):
    original_ids = [s.prompt_id for s in themed_dataset.samples]
    stratified_sample(themed_dataset, n=5, tag_prefix="theme")
    assert [s.prompt_id for s in themed_dataset.samples] == original_ids


# ---------------------------------------------------------------------------
# Realistic samples — conftest fixtures
#
# main_dataset   : 2 samples (simple_sample + multi_turn_sample), subset="main"
# consensus_dataset : 1 sample with ideal_completions_data, subset="consensus"
# ---------------------------------------------------------------------------


def test_sample_dataset_with_realistic_samples_returns_subset_label(main_dataset):
    result = sample_dataset(main_dataset, n=1)
    assert result.subset == "main"


def test_sample_dataset_preserves_canary_on_sampled_items(main_dataset):
    result = sample_dataset(main_dataset, n=2)
    for sample in result.samples:
        assert sample.canary is not None


def test_sample_dataset_multi_turn_sample_preserved_intact(main_dataset):
    # main_dataset contains a multi-turn sample; sampling all should keep its turn count
    result = sample_dataset(main_dataset, n=2, seed=0)
    multi_turn = [s for s in result.samples if len(s.prompt) > 1]
    assert len(multi_turn) == 1
    assert len(multi_turn[0].prompt) == 3


def test_sample_dataset_ideal_completions_data_preserved(consensus_dataset):
    result = sample_dataset(consensus_dataset, n=1)
    assert result.samples[0].ideal_completions_data is not None


def test_stratified_sample_on_consensus_subset_preserves_label(consensus_dataset):
    result = stratified_sample(consensus_dataset, n=1, tag_prefix="theme")
    assert result.subset == "consensus"


def test_stratified_sample_on_main_dataset_respects_n(main_dataset):
    result = stratified_sample(main_dataset, n=1, tag_prefix="theme")
    assert len(result) <= 1
