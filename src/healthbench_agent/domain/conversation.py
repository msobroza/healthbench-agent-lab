"""Primitive conversation types and the rich domain Conversation model.

Aligned with simple-evals types.py: Message / MessageList are plain dicts.
ConversationMetadata and Conversation are the richer project-level model.

Nothing in this module imports from the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .rubric import RubricItem

# ---------------------------------------------------------------------------
# Primitive message types (mirrors simple-evals Message / MessageList)
# ---------------------------------------------------------------------------

Message = dict[str, Any]  # keys: "role" (str), "content" (str | list)
MessageList = list[Message]  # ordered turns forming a conversation


# ---------------------------------------------------------------------------
# Rich domain conversation model
# ---------------------------------------------------------------------------


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
        turns: Ordered conversation turns as a MessageList (role + content dicts).
        rubric_items: Graded rubric items used to score an agent response.
        metadata: Stratification and provenance metadata.
        example_tags: Dataset-level tags for stratified scoring (themes, axes).
    """

    conversation_id: str
    language: str
    specialty: str
    user_persona: Literal["patient", "healthcare professional"]
    turns: MessageList
    rubric_items: list[RubricItem]
    metadata: ConversationMetadata
    example_tags: list[str] = field(default_factory=list)
