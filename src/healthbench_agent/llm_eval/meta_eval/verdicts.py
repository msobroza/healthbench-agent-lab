"""Verdict DataFrame row builders for the meta-eval runner.

The runner uses these helpers to convert each ``(sample, rubric,
verdict)`` triple into a row dict and to flatten k-pass grading flows
into a single flat list of rows.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.meta_evaluation import LabelledSample
from healthbench_agent.domain.rubric import RubricItem


def _row(
    sample: LabelledSample,
    rubric: RubricItem,
    verdict: CriterionVerdict,
    k: int,
    gold_source: str,
    expected_met: bool,
    dimension_extractor: Callable[[RubricItem], str | None],
) -> dict[str, Any]:
    """Build one verdict-row dict matching the run_meta_eval schema.

    Args:
        sample: Source labelled sample.
        rubric: Rubric item being graded.
        verdict: Judge verdict for this rubric on this sample.
        k: Index of the k-pass that produced the verdict (1-indexed).
        gold_source: One of ``ideal_completion`` / ``example_meets`` /
            ``example_fails``.
        expected_met: Ground-truth label for this row.
        dimension_extractor: Maps a rubric item to its dimension tag.

    Returns:
        Dict of column values, matching the verdict DataFrame schema.
    """
    return {
        "prompt_id": sample.prompt_id,
        "criterion_id": rubric.criterion_id,
        "criterion": rubric.criterion[:200],
        "rubric_key": rubric.criterion_id or rubric.criterion,
        "dimension": dimension_extractor(rubric),
        "points": rubric.points,
        "sample_k": k,
        "gold_source": gold_source,
        "observed_met": bool(verdict.criteria_met),
        "expected_met": bool(expected_met),
        "specialty": sample.specialty,
        "language": sample.language,
        "metadata_json": json.dumps(sample.metadata),
    }


def _build_verdict_rows(
    judge: JudgeGrader,
    samples: list[LabelledSample],
    dimension_extractor: Callable[[RubricItem], str | None],
    k: int,
) -> list[dict[str, Any]]:
    """Run one k-pass over each (sample, flow) combination and emit verdict rows.

    For each sample, runs three independent grading flows when the
    relevant data is present:

    * ``ideal_completion`` — grade the gold response against all non-zero
      point rubrics
    * ``example_meets`` — grade each rubric's adversarial known-good
      example (expected True)
    * ``example_fails`` — grade each rubric's adversarial known-bad
      example (expected False)

    Args:
        judge: Grader to invoke for every flow.
        samples: Filter-surviving labelled samples.
        dimension_extractor: Maps a rubric item to its optional dimension tag.
        k: Pass index stamped on every emitted row's ``sample_k`` column.
            Callers that need multiple passes loop externally and invoke
            this helper once per ``k`` (so each call can be dispatched to
            its own worker).

    Returns:
        Flat list of row dicts, ready to feed into ``pd.DataFrame``.
    """
    rows: list[dict[str, Any]] = []
    for sample in samples:
        # Sample-level flow: grade the gold response.
        if sample.gold_response is not None:
            gold_rubrics = [r for r in sample.rubrics if r.points != 0]
            if gold_rubrics:
                gold_turn: dict[str, Any] = {
                    "role": "assistant",
                    "content": sample.gold_response,
                }
                conversation = sample.prompt + [gold_turn]
                verdicts = judge.grade(conversation, gold_rubrics)
                for rubric, verdict in zip(gold_rubrics, verdicts, strict=True):
                    rows.append(
                        _row(
                            sample,
                            rubric,
                            verdict,
                            k,
                            "ideal_completion",
                            expected_met=rubric.points > 0,
                            dimension_extractor=dimension_extractor,
                        )
                    )
        # Adversarial flows: grade the known-good and known-bad examples.
        for rubric in sample.rubrics:
            if rubric.example_meets is not None:
                meets_turn: dict[str, Any] = {
                    "role": "assistant",
                    "content": rubric.example_meets,
                }
                conversation = sample.prompt + [meets_turn]
                [verdict] = judge.grade(conversation, [rubric])
                rows.append(
                    _row(
                        sample,
                        rubric,
                        verdict,
                        k,
                        "example_meets",
                        expected_met=True,
                        dimension_extractor=dimension_extractor,
                    )
                )
            if rubric.example_fails is not None:
                fails_turn: dict[str, Any] = {
                    "role": "assistant",
                    "content": rubric.example_fails,
                }
                conversation = sample.prompt + [fails_turn]
                [verdict] = judge.grade(conversation, [rubric])
                rows.append(
                    _row(
                        sample,
                        rubric,
                        verdict,
                        k,
                        "example_fails",
                        expected_met=False,
                        dimension_extractor=dimension_extractor,
                    )
                )
    return rows


_VERDICT_COLUMNS: list[str] = [
    "prompt_id",
    "criterion_id",
    "criterion",
    "rubric_key",
    "dimension",
    "points",
    "sample_k",
    "gold_source",
    "observed_met",
    "expected_met",
    "specialty",
    "language",
    "metadata_json",
]
