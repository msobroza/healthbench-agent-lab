"""Tests for healthbench_agent.analysis.utils.

Covers DEFAULT_PERCENTILES, series_stats, parse_tag_prefixes, and save_csv
using the ZOMBIES heuristic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from healthbench_agent.analysis.utils import (
    DEFAULT_PERCENTILES,
    parse_tag_prefixes,
    save_csv,
    series_stats,
)

# ---------------------------------------------------------------------------
# DEFAULT_PERCENTILES
# ---------------------------------------------------------------------------


def test_default_percentiles_has_five_levels():
    assert len(DEFAULT_PERCENTILES) == 5


def test_default_percentiles_are_between_zero_and_one():
    assert all(0.0 < p < 1.0 for p in DEFAULT_PERCENTILES)


def test_default_percentiles_includes_median():
    assert 0.50 in DEFAULT_PERCENTILES


# ---------------------------------------------------------------------------
# series_stats
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_series():
    return pd.Series([0.0, 0.25, 0.50, 0.75, 1.0])


def test_series_stats_returns_dict(uniform_series):
    result = series_stats(uniform_series)
    assert isinstance(result, dict)


def test_series_stats_contains_count(uniform_series):
    result = series_stats(uniform_series)
    assert "count" in result
    assert result["count"] == pytest.approx(5.0)


def test_series_stats_contains_mean(uniform_series):
    result = series_stats(uniform_series)
    assert "mean" in result
    assert result["mean"] == pytest.approx(0.5)


def test_series_stats_contains_min_and_max(uniform_series):
    result = series_stats(uniform_series)
    assert result["min"] == pytest.approx(0.0)
    assert result["max"] == pytest.approx(1.0)


def test_series_stats_default_percentiles_keys_present(uniform_series):
    result = series_stats(uniform_series)
    for p in DEFAULT_PERCENTILES:
        label = f"{int(p * 100)}%" if p * 100 == int(p * 100) else f"{p * 100}%"
        assert label in result
    # std + count + mean + min + max + 5 percentiles = 9
    assert len(result) >= 7


def test_series_stats_custom_percentiles_included():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = series_stats(s, percentiles=[0.10, 0.90])
    assert "10%" in result
    assert "90%" in result


def test_series_stats_single_element_series():
    s = pd.Series([42.0])
    result = series_stats(s)
    assert result["count"] == pytest.approx(1.0)
    assert result["mean"] == pytest.approx(42.0)


def test_series_stats_all_same_values():
    s = pd.Series([7.0, 7.0, 7.0])
    result = series_stats(s)
    assert result["std"] == pytest.approx(0.0)
    assert result["min"] == pytest.approx(7.0)
    assert result["max"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# parse_tag_prefixes
# ---------------------------------------------------------------------------


def test_parse_tag_prefixes_empty_list_returns_empty_dict():
    assert parse_tag_prefixes([]) == {}


def test_parse_tag_prefixes_single_prefixed_tag():
    result = parse_tag_prefixes(["theme:cardiology"])
    assert result == {"theme": {"cardiology": 1}}


def test_parse_tag_prefixes_counts_repeated_values():
    result = parse_tag_prefixes(["theme:cardiology", "theme:cardiology", "theme:oncology"])
    assert result["theme"]["cardiology"] == 2
    assert result["theme"]["oncology"] == 1


def test_parse_tag_prefixes_tag_without_prefix_goes_to_no_prefix():
    result = parse_tag_prefixes(["emergency_referral"])
    assert "_no_prefix" in result
    assert result["_no_prefix"]["emergency_referral"] == 1


def test_parse_tag_prefixes_mixed_tags():
    tags = ["theme:cardiology", "plain_tag", "axis:accuracy"]
    result = parse_tag_prefixes(tags)
    assert "theme" in result
    assert "axis" in result
    assert "_no_prefix" in result


def test_parse_tag_prefixes_multiple_prefixes_stay_separate():
    tags = ["theme:cardiology", "axis:accuracy", "theme:oncology"]
    result = parse_tag_prefixes(tags)
    assert set(result) == {"theme", "axis"}
    assert result["theme"] == {"cardiology": 1, "oncology": 1}
    assert result["axis"] == {"accuracy": 1}


def test_parse_tag_prefixes_value_with_colon_space_only_splits_on_first():
    result = parse_tag_prefixes(["key:value: extra"])
    assert result == {"key": {"value: extra": 1}}


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------


def test_save_csv_creates_file(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "output.csv"
    save_csv(df, path)
    assert path.exists()


def test_save_csv_creates_parent_directories(tmp_path):
    df = pd.DataFrame({"x": [1]})
    path = tmp_path / "nested" / "dir" / "result.csv"
    save_csv(df, path)
    assert path.exists()


def test_save_csv_written_content_is_readable(tmp_path):
    df = pd.DataFrame({"score": [0.5, 1.0]})
    path = tmp_path / "scores.csv"
    save_csv(df, path)
    loaded = pd.read_csv(path, index_col=0)
    assert list(loaded["score"]) == pytest.approx([0.5, 1.0])
