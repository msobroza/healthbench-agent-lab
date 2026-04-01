"""healthbench_agent — HealthBench agent toolkit.

Public API re-exported from subpackages so callers import from the package root:

    from healthbench_agent import HealthBenchDataset, SamplerBase, calculate_score
    from healthbench_agent import download_dataset, load_dataset
    from healthbench_agent import sample_dataset, stratified_sample
    from healthbench_agent import run_all, run_category, run_one

Package structure:
    domain/    — pure types and scoring (no I/O, no external deps)
    dataset/   — download, load, and split HealthBench JSONL files
    analysis/  — registered analysis functions and registry runners
"""

# ---------------------------------------------------------------------------
# Domain — types
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Analysis — registry and runners
# ---------------------------------------------------------------------------
from .analysis import (
    AnalysisEntry,
    register_analysis,
    run_all,
    run_category,
    run_one,
)

# ---------------------------------------------------------------------------
# Dataset — download, load, and split
# ---------------------------------------------------------------------------
from .dataset import (
    DATASET_FILENAMES,
    DATASET_URLS,
    DEFAULT_DATA_DIR,
    download_all_datasets,
    download_dataset,
    load_dataset,
    sample_dataset,
    stratified_sample,
)

# ---------------------------------------------------------------------------
# Domain — scoring
# ---------------------------------------------------------------------------
from .domain import (
    Conversation,
    ConversationMetadata,
    CriterionVerdict,
    DatasetSubset,
    Eval,
    EvalResult,
    HealthBenchDataset,
    HealthBenchSample,
    Message,
    MessageList,
    RubricItem,
    SamplerBase,
    SamplerResponse,
    SingleEvalResult,
    aggregate_scores,
    calculate_score,
    clip_score,
    stratified_scores,
)

__all__ = [
    # conversation primitives
    "Message",
    "MessageList",
    # conversation domain
    "ConversationMetadata",
    "Conversation",
    # rubric
    "RubricItem",
    # sampler
    "SamplerBase",
    "SamplerResponse",
    # dataset domain types
    "DatasetSubset",
    "HealthBenchSample",
    "HealthBenchDataset",
    # evaluation types
    "CriterionVerdict",
    "SingleEvalResult",
    "EvalResult",
    "Eval",
    # scoring
    "calculate_score",
    "clip_score",
    "aggregate_scores",
    "stratified_scores",
    # dataset — download, load, and split
    "download_dataset",
    "download_all_datasets",
    "load_dataset",
    "sample_dataset",
    "stratified_sample",
    "DATASET_URLS",
    "DATASET_FILENAMES",
    "DEFAULT_DATA_DIR",
    # analysis
    "register_analysis",
    "run_one",
    "run_category",
    "run_all",
    "AnalysisEntry",
]
