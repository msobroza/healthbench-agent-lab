"""Tests for healthbench_agent.dataset.loader — download functions.

Covers download_dataset and download_all_datasets using ZOMBIES heuristic.
No real network calls — urllib.request.urlretrieve is patched throughout.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from healthbench_agent.dataset.loader import (
    DATASET_FILENAMES,
    DATASET_URLS,
    DEFAULT_DATA_DIR,
    download_all_datasets,
    download_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_dataset_urls_has_three_subsets():
    assert set(DATASET_URLS) == {"main", "hard", "consensus"}


def test_dataset_filenames_has_three_subsets():
    assert set(DATASET_FILENAMES) == {"main", "hard", "consensus"}


def test_default_data_dir_ends_with_data_healthbench():
    assert DEFAULT_DATA_DIR.parts[-2:] == ("data", "healthbench")


def test_default_data_dir_is_inside_project(tmp_path):
    # Must be absolute and not inside the package source directory
    assert DEFAULT_DATA_DIR.is_absolute()
    assert "src" not in DEFAULT_DATA_DIR.parts


# ---------------------------------------------------------------------------
# download_dataset — invalid input
# ---------------------------------------------------------------------------


def test_download_dataset_unknown_subset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown subset"):
        download_dataset(subset="unknown")


# ---------------------------------------------------------------------------
# download_dataset — skip when file exists
# ---------------------------------------------------------------------------


def test_download_dataset_skips_when_file_exists(tmp_path):
    destination = tmp_path / DATASET_FILENAMES["main"]
    destination.write_text("existing content")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        result = download_dataset(subset="main", data_dir=tmp_path)

    mock_retrieve.assert_not_called()
    assert result == destination


# ---------------------------------------------------------------------------
# download_dataset — force re-download
# ---------------------------------------------------------------------------


def test_download_dataset_force_overwrites_existing_file(tmp_path):
    destination = tmp_path / DATASET_FILENAMES["main"]
    destination.write_text("old content")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("new content")
        result = download_dataset(subset="main", data_dir=tmp_path, force=True)

    mock_retrieve.assert_called_once()
    assert result == destination


# ---------------------------------------------------------------------------
# download_dataset — creates data_dir when absent
# ---------------------------------------------------------------------------


def test_download_dataset_creates_data_dir_if_missing(tmp_path):
    new_dir = tmp_path / "nested" / "dir"
    assert not new_dir.exists()

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        download_dataset(subset="main", data_dir=new_dir)

    assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# download_dataset — returns correct path
# ---------------------------------------------------------------------------


def test_download_dataset_returns_path_to_file(tmp_path):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        result = download_dataset(subset="hard", data_dir=tmp_path)

    assert result == tmp_path / DATASET_FILENAMES["hard"]


# ---------------------------------------------------------------------------
# download_dataset — correct URL used per subset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subset", ["main", "hard", "consensus"])
def test_download_dataset_uses_correct_url_for_subset(tmp_path, subset):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        download_dataset(subset=subset, data_dir=tmp_path)

    called_url = mock_retrieve.call_args[0][0]
    assert called_url == DATASET_URLS[subset]


# ---------------------------------------------------------------------------
# download_dataset — URLError propagates
# ---------------------------------------------------------------------------


def test_download_dataset_propagates_url_error(tmp_path):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = urllib.error.URLError("connection refused")
        with pytest.raises(urllib.error.URLError):
            download_dataset(subset="main", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# download_all_datasets
# ---------------------------------------------------------------------------


def test_download_all_datasets_returns_all_three_subsets(tmp_path):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        results = download_all_datasets(data_dir=tmp_path)

    assert set(results) == {"main", "hard", "consensus"}


def test_download_all_datasets_paths_match_filenames(tmp_path):
    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        results = download_all_datasets(data_dir=tmp_path)

    for subset, path in results.items():
        assert path == tmp_path / DATASET_FILENAMES[subset]


def test_download_all_datasets_skips_existing_without_force(tmp_path):
    existing = tmp_path / DATASET_FILENAMES["main"]
    existing.write_text("cached")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("data")
        download_all_datasets(data_dir=tmp_path)

    # Only hard and consensus should have been fetched
    assert mock_retrieve.call_count == 2


def test_download_all_datasets_force_downloads_all(tmp_path):
    for name in DATASET_FILENAMES.values():
        (tmp_path / name).write_text("old")

    with patch("urllib.request.urlretrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, dest: Path(dest).write_text("new")
        download_all_datasets(data_dir=tmp_path, force=True)

    assert mock_retrieve.call_count == 3
