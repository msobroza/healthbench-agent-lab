"""HealthBench domain subpackage — pure types and scoring, no I/O.

All types and functions are re-exported here so callers can import from
either the submodule or the domain package root:

    from healthbench_agent.domain import RubricItem, calculate_score
    from healthbench_agent.domain.rubric import RubricItem

Dependency order (each module only imports from modules above it):
    rubric.py
    conversation.py  → rubric
    sampler.py       → conversation
    evaluation.py    → conversation, sampler
    dataset.py       → rubric, conversation
    scoring.py       → evaluation, rubric
"""

from .conversation import Conversation, ConversationMetadata, Message, MessageList
from .dataset import DatasetSubset, HealthBenchDataset, HealthBenchSample
from .evaluation import CriterionVerdict, Eval, EvalResult, SingleEvalResult
from .rubric import RubricItem
from .sampler import SamplerBase, SamplerResponse
from .scoring import aggregate_scores, calculate_score, clip_score, stratified_scores

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
    # evaluation
    "CriterionVerdict",
    "SingleEvalResult",
    "EvalResult",
    "Eval",
    # dataset
    "DatasetSubset",
    "HealthBenchSample",
    "HealthBenchDataset",
    # scoring
    "calculate_score",
    "clip_score",
    "aggregate_scores",
    "stratified_scores",
]
