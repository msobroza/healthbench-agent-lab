"""Shared domain models for the HealthBench agent project.

All modules (agents, evaluation, analysis) import from here.
Nothing in this module imports from the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """A single turn in a health conversation.

    Attributes:
        role: Speaker identity — either the patient/user or the AI assistant.
        content: Raw text of the turn.
        turn_number: 1-based position in the conversation sequence.
    """

    role: Literal["user", "assistant"]
    content: str
    turn_number: int


@dataclass
class RubricCriterion:
    """One graded criterion within a HealthBench rubric.

    Attributes:
        criterion_id: Unique identifier used to match verdicts to criteria.
        description: Human-readable statement of what the criterion checks.
        weight: Points awarded (positive) or deducted (negative) when met.
            Range: [-10, 10]. Emergency/safety criteria carry the highest weights.
        category: Theme or axis label this criterion belongs to.
        example_meets: Optional illustrative response that satisfies this criterion.
        example_fails: Optional illustrative response that violates this criterion.
    """

    criterion_id: str
    description: str
    weight: float
    category: str
    example_meets: str = ""
    example_fails: str = ""


@dataclass
class Rubric:
    """Complete rubric for a single HealthBench conversation.

    Attributes:
        criteria: All graded criteria for this conversation.
        max_score: Sum of max(0, weight) across all criteria — the denominator
            in the HealthBench scoring formula.
    """

    criteria: list[RubricCriterion]
    max_score: float


@dataclass
class ConversationMetadata:
    """Stratification and provenance metadata attached to a conversation.

    Attributes:
        difficulty: Subset label — 'standard' (consensus) or 'hard'.
        variant: Evaluation variant — 'consensus' or 'hard'.
        health_literacy_level: Target audience literacy level.
        clinical_urgency: Triage category for the presented scenario.
        language_family: Linguistic family of the conversation language.
        sub_specialty: Narrower medical sub-specialty within the specialty field.
        cultural_context: Cultural or regional context affecting clinical norms.
        validator_specialties: Physician specialties of the validators who rated
            this conversation.
        adversarial_tested: Whether the conversation was stress-tested with
            adversarial prompts.
    """

    difficulty: Literal["standard", "hard"]
    variant: Literal["consensus", "hard"]
    health_literacy_level: Literal["low", "medium", "high", "professional"]
    clinical_urgency: Literal["routine", "urgent", "emergency"]
    language_family: str = ""
    sub_specialty: str = ""
    cultural_context: str = ""
    validator_specialties: list[str] = field(default_factory=list)
    adversarial_tested: bool = False


@dataclass
class Conversation:
    """A complete HealthBench conversation with its rubric and metadata.

    Attributes:
        conversation_id: Globally unique identifier for this conversation.
        language: ISO 639-1 language code (e.g. 'en', 'fr', 'zh').
        specialty: Primary medical specialty (e.g. 'cardiology', 'emergency').
        user_persona: Role of the human turn author.
        turns: Ordered list of conversation turns, alternating user/assistant.
        rubric: Graded rubric used to score an agent response to this conversation.
        metadata: Stratification and provenance metadata.
    """

    conversation_id: str
    language: str
    specialty: str
    user_persona: Literal["patient", "healthcare professional"]
    turns: list[ConversationTurn]
    rubric: Rubric
    metadata: ConversationMetadata


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


@dataclass
class CriterionVerdict:
    """LLM-judge verdict for a single rubric criterion.

    Attributes:
        criterion_id: Matches `RubricCriterion.criterion_id`.
        met: Whether the criterion was satisfied by the agent response.
        confidence: Majority-vote confidence from multi-sample scoring.
            Range: [0.0, 1.0]. Defaults to 1.0 for single-sample judges.
    """

    criterion_id: str
    met: bool
    confidence: float = 1.0


@dataclass
class EvalResult:
    """Evaluation outcome for one agent response to one conversation.

    Attributes:
        conversation_id: Links back to `Conversation.conversation_id`.
        agent_name: Identifier of the agent variant that produced the response.
        prompt_version: Prompt YAML filename (e.g. 'v1_baseline') used in this run.
        model: Model identifier (e.g. 'gemini-2.0-flash').
        verdicts: Per-criterion LLM-judge verdicts.
        score: HealthBench score in (-inf, 1.0]. Can be negative before clipping
            when penalty criteria dominate.
        theme_scores: Per-theme scores keyed by theme label.
        axis_scores: Per-axis scores keyed by axis label.
    """

    conversation_id: str
    agent_name: str
    prompt_version: str
    model: str
    verdicts: list[CriterionVerdict]
    score: float
    theme_scores: dict[str, float] = field(default_factory=dict)
    axis_scores: dict[str, float] = field(default_factory=dict)
