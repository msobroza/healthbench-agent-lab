"""HealthBench dataset subpackage — download, load, and split utilities.

Public API:
    from healthbench_agent.dataset import download_dataset, download_all_datasets
    from healthbench_agent.dataset import load_dataset
    from healthbench_agent.dataset import sample_dataset, stratified_sample
    from healthbench_agent.dataset import DATASET_URLS, DATASET_FILENAMES, DEFAULT_DATA_DIR
"""

from .loader import (
    DATASET_FILENAMES,
    DATASET_URLS,
    DEFAULT_DATA_DIR,
    download_all_datasets,
    download_dataset,
    load_dataset,
)
from .split_utils import sample_dataset, stratified_sample

__all__ = [
    "DATASET_URLS",
    "DATASET_FILENAMES",
    "DEFAULT_DATA_DIR",
    "download_dataset",
    "download_all_datasets",
    "load_dataset",
    "sample_dataset",
    "stratified_sample",
]
