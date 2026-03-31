"""Dataset splitting and sampling utilities for HealthBench.

Provides deterministic random sampling and stratified sampling by tag prefix,
supporting reproducible evaluation runs across agent variants.

Import chain: healthbench_agent.domain.dataset only.
No imports from agents/, evaluation/, or analysis/.
"""

from __future__ import annotations

import logging

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from ..domain.dataset import HealthBenchDataset, HealthBenchSample

logger = logging.getLogger(__name__)


def sample_dataset(
    dataset: HealthBenchDataset,
    n: int,
    seed: int = 0,
) -> HealthBenchDataset:
    """Draw a random sample of n samples from a dataset without replacement.

    Returns all samples when n >= len(dataset).

    Args:
        dataset: Source dataset to sample from.
        n: Number of samples to draw.
        seed: Random seed for reproducibility. Defaults to 0.

    Returns:
        A new HealthBenchDataset with the sampled subset.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    k = min(n, len(dataset.samples))
    if k == 0:
        return HealthBenchDataset(
            subset=dataset.subset, samples=[], source_path=dataset.source_path
        )

    sampled: list[HealthBenchSample] = resample(
        dataset.samples, replace=False, n_samples=k, random_state=seed
    )

    logger.info(
        "Sampled %d/%d samples from subset %r (seed=%d)",
        k, len(dataset.samples), dataset.subset, seed,
    )
    return HealthBenchDataset(
        subset=dataset.subset, samples=sampled, source_path=dataset.source_path
    )


def stratified_sample(
    dataset: HealthBenchDataset,
    n: int,
    tag_prefix: str,
    seed: int = 0,
) -> HealthBenchDataset:
    """Sample n items proportionally across strata defined by a tag prefix.

    Uses sklearn stratified splitting so each stratum is represented in
    proportion to its share of the dataset. Samples without the given prefix
    are grouped under '_no_prefix'.

    Returns all samples when n >= len(dataset).

    Args:
        dataset: Source dataset to sample from.
        n: Total number of samples to draw.
        tag_prefix: Tag prefix used to define strata, e.g. 'theme' or 'axis'.
            Tags must follow the 'prefix: value' convention.
        seed: Random seed for reproducibility. Defaults to 0.

    Returns:
        A new HealthBenchDataset containing the stratified sample.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    samples = dataset.samples
    total = len(samples)

    if n == 0 or total == 0:
        return HealthBenchDataset(
            subset=dataset.subset, samples=[], source_path=dataset.source_path
        )

    if n >= total:
        return HealthBenchDataset(
            subset=dataset.subset, samples=list(samples), source_path=dataset.source_path
        )

    labels = [_extract_stratum(s.example_tags, tag_prefix) for s in samples]
    _, sampled = train_test_split(
        samples, test_size=n, stratify=labels, random_state=seed
    )

    logger.info(
        "Stratified sample of %d/%d from subset %r by prefix %r (seed=%d)",
        len(sampled), total, dataset.subset, tag_prefix, seed,
    )
    return HealthBenchDataset(
        subset=dataset.subset, samples=sampled, source_path=dataset.source_path
    )


def _extract_stratum(tags: list[str], prefix: str) -> str:
    """Return the first value matching the given tag prefix, or '_no_prefix'.

    Args:
        tags: List of raw tag strings from a HealthBenchSample.
        prefix: Tag prefix to match, e.g. 'theme'.

    Returns:
        The tag value for the given prefix, or '_no_prefix' if absent.
    """
    needle = f"{prefix}: "
    for tag in tags:
        if tag.startswith(needle):
            return tag[len(needle):].strip()
    return "_no_prefix"
