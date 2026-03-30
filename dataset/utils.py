"""Dataset download and loading utilities for HealthBench.

Provides download_dataset() to fetch the three HealthBench JSONL subsets
(main, hard, consensus) from the OpenAI public blob storage, and
load_dataset() to deserialize them into a HealthBenchDataset.

Dataset URLs mirror the constants in simple-evals healthbench_eval.py.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from healthbench_agent.data_models import (
    DatasetSubset,
    HealthBenchDataset,
    HealthBenchSample,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset URLs (mirrors simple-evals healthbench_eval.py INPUT_PATH constants)
# ---------------------------------------------------------------------------

_DATASET_URLS: dict[str, str] = {
    "main": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl",
    "consensus": "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl",
}

_DATASET_FILENAMES: dict[str, str] = {
    "main": "healthbench.jsonl",
    "hard": "healthbench_hard.jsonl",
    "consensus": "healthbench_consensus.jsonl",
}

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "healthbench"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_dataset(
    subset: DatasetSubset = "main",
    data_dir: Path = _DEFAULT_DATA_DIR,
    force: bool = False,
) -> Path:
    """Download a HealthBench JSONL subset from OpenAI public blob storage.

    Skips the download if the file already exists, unless force=True.

    Args:
        subset: Which subset to download — 'main' (5,000 samples),
            'hard' (1,000 samples), or 'consensus' (3,671 samples).
        data_dir: Local directory to write the file into.
            Defaults to data/healthbench/ relative to the project root.
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded JSONL file.

    Raises:
        ValueError: If subset is not one of 'main', 'hard', 'consensus'.
        urllib.error.URLError: If the download fails due to a network error.
    """
    if subset not in _DATASET_URLS:
        raise ValueError(
            f"Unknown subset {subset!r}. Choose from: {list(_DATASET_URLS)}"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / _DATASET_FILENAMES[subset]

    if destination.exists() and not force:
        logger.info("Already exists, skipping download: %s", destination)
        return destination

    url = _DATASET_URLS[subset]
    logger.info("Downloading %r subset from %s", subset, url)
    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.URLError:
        logger.exception("Failed to download %r subset from %s", subset, url)
        raise
    logger.info("Saved to %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def download_all_datasets(
    data_dir: Path = _DEFAULT_DATA_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download all three HealthBench subsets.

    Args:
        data_dir: Local directory to write the files into.
        force: Re-download even if files already exist.

    Returns:
        Mapping of subset name to the downloaded file path.
    """
    return {
        subset: download_dataset(subset=subset, data_dir=data_dir, force=force)
        for subset in _DATASET_URLS
    }


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_dataset(
    subset: DatasetSubset = "main",
    data_dir: Path = _DEFAULT_DATA_DIR,
) -> HealthBenchDataset:
    """Load a HealthBench subset from a local JSONL file into a HealthBenchDataset.

    Downloads the file first if it does not exist locally.

    Args:
        subset: Which subset to load — 'main', 'hard', or 'consensus'.
        data_dir: Directory containing the JSONL files.

    Returns:
        HealthBenchDataset with all samples deserialized from the JSONL file.
    """
    path = data_dir / _DATASET_FILENAMES[subset]
    if not path.exists():
        download_dataset(subset=subset, data_dir=data_dir)

    samples: list[HealthBenchSample] = []
    with path.open() as file:
        for line in file:
            samples.append(HealthBenchSample.from_dict(json.loads(line)))

    logger.info("Loaded %d samples from %s", len(samples), path)
    return HealthBenchDataset(
        subset=subset,
        samples=samples,
        source_path=str(path.resolve()),
    )
