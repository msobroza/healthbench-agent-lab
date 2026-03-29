"""healthbench_agent — shared types and scoring for the HealthBench agent project.

Public API re-exported from submodules so callers import from the package root:

    from healthbench_agent import Conversation, criterion_score

Types (from .types):
    ConversationTurn, RubricCriterion, Rubric, ConversationMetadata,
    Conversation, CriterionVerdict, EvalResult

Scoring (from .scoring):
    criterion_score, clip_score, aggregate_scores, stratified_scores
"""

from .scoring import aggregate_scores, clip_score, criterion_score, stratified_scores
from .models import (
    Conversation,
    ConversationMetadata,
    ConversationTurn,
    CriterionVerdict,
    EvalResult,
    Rubric,
    RubricCriterion,
)

__all__ = [
    # types
    "Conversation",
    "ConversationMetadata",
    "ConversationTurn",
    "CriterionVerdict",
    "EvalResult",
    "Rubric",
    "RubricCriterion",
    # scoring
    "aggregate_scores",
    "clip_score",
    "criterion_score",
    "stratified_scores",
]
