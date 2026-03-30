"""Shared domain models for the HealthBench agent project.

Aligned with the simple-evals type system (healthbench_eval.py + types.py):
  - Message / MessageList — primitive conversation turn types
  - SamplerBase / SamplerResponse — uniform model abstraction
  - RubricItem — single graded criterion (criterion, points, tags)
  - Conversation / ConversationMetadata — rich domain conversation model
  - HealthBenchSample — one JSONL row from the HealthBench dataset
  - HealthBenchDataset — loaded dataset with subset metadata
  - CriterionVerdict — typed LLM-judge verdict per criterion
  - SingleEvalResult / EvalResult / Eval — per-sample and aggregate eval results

Nothing in this module imports from the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Primitive message types (mirrors simple-evals Message / MessageList)
# ---------------------------------------------------------------------------

Message = dict[str, Any]     # keys: "role" (str), "content" (str | list)
MessageList = list[Message]  # ordered turns forming a conversation


# ---------------------------------------------------------------------------
# Sampler abstraction (mirrors simple-evals SamplerBase / SamplerResponse)
# ---------------------------------------------------------------------------


@dataclass
class SamplerResponse:
    """Response returned by a model sampler.

    Attributes:
        response_text: Final text output from the model.
        actual_queried_message_list: The exact message list sent to the model,
            after any prompt assembly or injection.
        response_metadata: Provider-specific metadata (token counts, finish
            reason, latency, etc.).
    """

    response_text: str
    actual_queried_message_list: MessageList
    response_metadata: dict[str, Any]


class SamplerBase:
    """Abstract base class for a model sampler.

    Subclasses wrap a specific model provider (e.g. Gemini, OpenAI) and
    expose a uniform call interface used by all Eval implementations.
    """

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        """Sample a response from the model.

        Args:
            message_list: Conversation history to condition the response on.

        Returns:
            A SamplerResponse with the model output and request metadata.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HealthBench rubric item (mirrors simple-evals RubricItem)
# ---------------------------------------------------------------------------


@dataclass
class RubricItem:
    """One graded criterion within a HealthBench rubric.

    Mirrors simple-evals RubricItem with fields: criterion, points, tags.

    Attributes:
        criterion: Human-readable statement of what the criterion checks.
        points: Points awarded (positive) or deducted (negative) when met.
            Range: [-10, 10]. Emergency/safety criteria carry the highest weights.
        tags: Theme and axis labels this criterion belongs to (e.g. 'accuracy',
            'emergency_referral').
    """

    criterion: str
    points: float
    tags: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.points}] {self.criterion}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricItem:
        """Deserialize from a JSON-compatible dict.

        Args:
            data: Dict with keys 'criterion', 'points', 'tags'.

        Returns:
            A RubricItem instance.
        """
        return cls(
            criterion=data["criterion"],
            points=data["points"],
            tags=data["tags"],
        )


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


# ---------------------------------------------------------------------------
# HealthBench dataset sample and dataset (mirrors JSONL row schema from simple-evals)
# ---------------------------------------------------------------------------


@dataclass
class HealthBenchSample:
    """One sample loaded from a HealthBench JSONL dataset file.

    Field names match the JSONL keys used in simple-evals healthbench_eval.py.

    Attributes:
        prompt_id: Unique identifier for this prompt.
        prompt: Conversation history as a MessageList (role + content dicts).
        rubrics: Graded criteria used to score a response to this prompt.
        example_tags: Dataset-level tags for stratified scoring (themes, axes).
        ideal_completions_data: Physician ideal completion data when available,
            including the completion group and reference responses. None for
            samples without physician annotations.
    """

    prompt_id: str
    prompt: MessageList
    rubrics: list[RubricItem]
    example_tags: list[str]
    ideal_completions_data: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HealthBenchSample:
        """Deserialize from a JSONL row dict.

        Args:
            data: Parsed JSONL row with keys prompt_id, prompt, rubrics,
                example_tags, and optionally ideal_completions_data.

        Returns:
            A HealthBenchSample instance with rubrics deserialized as RubricItem.
        """
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            rubrics=[RubricItem.from_dict(r) for r in data["rubrics"]],
            example_tags=data["example_tags"],
            ideal_completions_data=data.get("ideal_completions_data"),
        )


@dataclass
class HealthBenchDataset:
    """A loaded HealthBench dataset subset with its samples and metadata.

    Attributes:
        subset: Which subset was loaded — 'main', 'hard', or 'consensus'.
        samples: All samples in the dataset, one per JSONL row.
        source_path: Absolute path to the JSONL file that was loaded.
    """

    subset: DatasetSubset
    samples: list[HealthBenchSample]
    source_path: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# ---------------------------------------------------------------------------
# Evaluation results (mirrors simple-evals SingleEvalResult / EvalResult)
# ---------------------------------------------------------------------------


@dataclass
class CriterionVerdict:
    """Typed LLM-judge verdict for a single rubric criterion.

    Field names mirror the grading dict keys used in simple-evals
    (criteria_met, explanation) for easy interop.

    Attributes:
        criterion: The criterion text that was graded.
        criteria_met: Whether the criterion was satisfied by the agent response.
        explanation: Grader explanation for the verdict.
        confidence: Majority-vote confidence from multi-sample scoring.
            Range: [0.0, 1.0]. Defaults to 1.0 for single-sample judges.
    """

    criterion: str
    criteria_met: bool
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class SingleEvalResult:
    """Evaluation outcome for a single conversation sample.

    Mirrors simple-evals SingleEvalResult. HealthBench-specific data such as
    rubric verdicts, prompt_id, and token usage is stored in
    example_level_metadata.

    Attributes:
        score: HealthBench score in (-inf, 1.0]. None if scoring failed.
            Can be negative when penalty criteria dominate.
        metrics: Per-label scores for stratified analysis keyed by tag
            (e.g. 'overall_score', 'accuracy', 'emergency_referral').
        html: Optional HTML rendering of this sample's evaluation.
        convo: The full conversation including the sampled assistant response.
        example_level_metadata: Per-sample data such as verdicts, prompt_id,
            completion_id, usage stats, and raw rubric grades.
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    html: str | None = None
    convo: MessageList | None = None
    example_level_metadata: dict[str, Any] | None = None


@dataclass
class EvalResult:
    """Aggregate result of running an evaluation over many conversations.

    Mirrors simple-evals EvalResult. One EvalResult is produced per
    agent × prompt-version × dataset-subset run.

    Attributes:
        score: Overall benchmark score (clipped mean across samples). None if
            no samples were scored.
        metrics: Aggregate stratified scores keyed by tag, plus bootstrap
            standard deviations and sample counts (e.g. 'accuracy:bootstrap_std',
            'accuracy:n_samples'). None when no stratification was requested.
        htmls: Per-sample HTML renderings, one per conversation evaluated.
        convos: Sampled conversations, one MessageList per conversation.
        metadata: Run-level metadata: agent_name, prompt_version, model,
            sample_size, dataset_subset, and per-sample metadata list.
    """

    score: float | None
    metrics: dict[str, float] | None
    htmls: list[str]
    convos: list[MessageList]
    metadata: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Eval base class (mirrors simple-evals Eval)
# ---------------------------------------------------------------------------


class Eval:
    """Abstract base class for a HealthBench evaluation.

    Subclasses implement __call__ to score a sampler against a dataset
    and return an aggregate EvalResult.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation against the provided sampler.

        Args:
            sampler: Model sampler to evaluate.

        Returns:
            Aggregate EvalResult with overall score and per-sample data.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataset subset type alias
# ---------------------------------------------------------------------------

DatasetSubset = Literal["main", "hard", "consensus"]
