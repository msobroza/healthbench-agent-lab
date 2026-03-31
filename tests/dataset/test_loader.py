"""Tests for healthbench_agent.dataset.loader.

Covers load_dataset using ZOMBIES heuristic.
No real network calls or disk reads — file I/O is patched throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from healthbench_agent.domain import HealthBenchDataset, HealthBenchSample
from healthbench_agent.dataset.loader import (
    DATASET_FILENAMES,
    DEFAULT_DATA_DIR,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_ROW: dict = {
    "prompt_id": "p001",
    "prompt": [{"role": "user", "content": "I have chest pain."}],
    "rubrics": [{"criterion": "Refer immediately", "points": 1.0, "tags": ["emergency"]}],
    "example_tags": ["cardiology"],
    "ideal_completions_data": None,
}


def _jsonl_lines(*rows) -> str:
    return "\n".join(json.dumps(r) for r in rows)


# ---------------------------------------------------------------------------
# load_dataset — file present, no download triggered
# ---------------------------------------------------------------------------


def test_load_dataset_does_not_download_when_file_exists(tmp_path):
    path = tmp_path / DATASET_FILENAMES["main"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    with patch("healthbench_agent.dataset.loader.download_dataset") as mock_dl:
        load_dataset(subset="main", data_dir=tmp_path)

    mock_dl.assert_not_called()


def test_load_dataset_triggers_download_when_file_absent(tmp_path):
    def fake_download(subset, data_dir):
        (data_dir / DATASET_FILENAMES[subset]).write_text(_jsonl_lines(_SAMPLE_ROW))

    with patch(
        "healthbench_agent.dataset.loader.download_dataset",
        side_effect=fake_download,
    ) as mock_dl:
        load_dataset(subset="main", data_dir=tmp_path)

    mock_dl.assert_called_once_with(subset="main", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# load_dataset — return type and contents
# ---------------------------------------------------------------------------


def test_load_dataset_returns_health_bench_dataset(tmp_path):
    path = tmp_path / DATASET_FILENAMES["main"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    result = load_dataset(subset="main", data_dir=tmp_path)

    assert isinstance(result, HealthBenchDataset)


def test_load_dataset_subset_label_matches_requested_subset(tmp_path):
    path = tmp_path / DATASET_FILENAMES["hard"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    result = load_dataset(subset="hard", data_dir=tmp_path)

    assert result.subset == "hard"


def test_load_dataset_sample_count_matches_jsonl_rows(tmp_path):
    path = tmp_path / DATASET_FILENAMES["main"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW, _SAMPLE_ROW))

    result = load_dataset(subset="main", data_dir=tmp_path)

    assert len(result) == 2


def test_load_dataset_deserializes_samples_as_health_bench_sample(tmp_path):
    path = tmp_path / DATASET_FILENAMES["main"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    result = load_dataset(subset="main", data_dir=tmp_path)

    assert all(isinstance(s, HealthBenchSample) for s in result)


def test_load_dataset_source_path_is_absolute(tmp_path):
    path = tmp_path / DATASET_FILENAMES["main"]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    result = load_dataset(subset="main", data_dir=tmp_path)

    assert Path(result.source_path).is_absolute()


# ---------------------------------------------------------------------------
# load_dataset — empty file
# ---------------------------------------------------------------------------


def test_load_dataset_empty_file_returns_empty_dataset(tmp_path):
    path = tmp_path / DATASET_FILENAMES["consensus"]
    path.write_text("")

    result = load_dataset(subset="consensus", data_dir=tmp_path)

    assert len(result) == 0


# ---------------------------------------------------------------------------
# load_dataset — all valid subsets accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subset", ["main", "hard", "consensus"])
def test_load_dataset_accepts_all_valid_subsets(tmp_path, subset):
    path = tmp_path / DATASET_FILENAMES[subset]
    path.write_text(_jsonl_lines(_SAMPLE_ROW))

    result = load_dataset(subset=subset, data_dir=tmp_path)

    assert result.subset == subset
