"""healthbench_agent — shared types and scoring for the HealthBench agent project.

Public API re-exported from submodules so callers import from the package root:

    from healthbench_agent import HealthBenchDataset, SamplerBase, calculate_score

Models (from .data_models):
    Message, MessageList,
    SamplerBase, SamplerResponse,
    RubricItem,
    ConversationMetadata, Conversation,
    HealthBenchSample, HealthBenchDataset,
    CriterionVerdict, SingleEvalResult, EvalResult, Eval,
    DatasetSubset

Scoring (from .scoring):
    calculate_score, clip_score, aggregate_scores, stratified_scores
"""

from .data_models import (
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
)
from .scoring import aggregate_scores, calculate_score, clip_score, stratified_scores

__all__ = [
    # primitive types
    "Message",
    "MessageList",
    # sampler
    "SamplerBase",
    "SamplerResponse",
    # rubric
    "RubricItem",
    # conversation
    "Conversation",
    "ConversationMetadata",
    # dataset
    "HealthBenchSample",
    "HealthBenchDataset",
    "DatasetSubset",
    # evaluation results
    "CriterionVerdict",
    "SingleEvalResult",
    "EvalResult",
    "Eval",
    # scoring
    "calculate_score",
    "clip_score",
    "aggregate_scores",
    "stratified_scores",
]
