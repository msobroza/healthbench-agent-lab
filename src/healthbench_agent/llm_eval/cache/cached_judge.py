"""``JudgeGrader`` proxy that consults a ``VerdictCache`` before delegating.

Kept in its own module so the pure on-disk store (:mod:`.store`) has no
dependency on the domain ``JudgeGrader`` ABC and can be reused by any
caller that only needs the key helper.

The ``get -> inner.grade -> put`` sequence inside :meth:`CachedJudgeGrader.grade`
is **not** atomic across threads: when two workers race on the same
``(conversation, rubric, k)`` key, both will miss, both will call
``inner.grade``, and the later ``put`` will overwrite the earlier one.
The race causes redundant LLM calls for that key but never stale or
torn reads, and cross-run deduplication — the primary purpose of the
cache — is preserved because ``put`` writes via ``os.replace``. In
practice the key space (samples × rubrics × k passes) dwarfs the worker
count, so collisions are rare and acceptable.
"""

from __future__ import annotations

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem

from .store import VerdictCache


class CachedJudgeGrader(JudgeGrader):
    """JudgeGrader proxy that consults a VerdictCache before delegating.

    Wraps any JudgeGrader and intercepts ``grade()`` to deduplicate
    verdicts. The cache key components that are constant across one
    k-pass (model fingerprint, prompt sha, k_index) are baked in at
    construction time so the proxy can compute the full key from just
    the (conversation, rubric_text) tuple — no introspection of the
    inner grader required.

    Thread safety:
        Safe to share across worker threads. The ``get -> inner.grade
        -> put`` sequence inside :meth:`grade` is **not** atomic per
        cache key — see the module docstring for the full contract.
        Cross-run deduplication is preserved; within-run duplicate
        ``inner.grade`` calls for the same key are possible but rare
        when the key space dwarfs the worker count.
    """

    def __init__(
        self,
        inner: JudgeGrader,
        cache: VerdictCache,
        model_fingerprint: str,
        prompt_sha: str,
        k_index: int,
    ) -> None:
        """Initialise the proxy.

        Args:
            inner: The underlying grader used for cache misses.
            cache: Verdict cache consulted before delegation.
            model_fingerprint: Identifier of the judge model (e.g.
                ``"openai/gpt-4.1@1.0"``) used as a cache key component.
            prompt_sha: Hash of the rendered judge prompt template.
            k_index: Index within a multi-sample majority vote; lets
                independent passes store distinct entries.
        """
        self.inner = inner
        self.cache = cache
        self.model_fingerprint = model_fingerprint
        self.prompt_sha = prompt_sha
        self.k_index = k_index

    def grade(
        self,
        conversation: MessageList,
        rubric_items: list[RubricItem],
    ) -> list[CriterionVerdict]:
        """Look up cached verdicts; batch the misses into one inner.grade() call.

        Args:
            conversation: Full conversation being graded.
            rubric_items: Rubric items to grade against, in input order.

        Returns:
            List of verdicts aligned with ``rubric_items``. Cached entries
            are served from the cache; misses are fetched in a single
            batched call to the inner grader and then persisted.
        """
        cached: dict[int, CriterionVerdict] = {}
        miss_indices: list[int] = []
        miss_items: list[RubricItem] = []
        for idx, item in enumerate(rubric_items):
            rubric_key = item.criterion_id or (
                f"{item.criterion}|points={item.points}|tags={tuple(item.tags)}"
            )
            key = self.cache.make_key(
                self.model_fingerprint,
                self.prompt_sha,
                conversation,
                rubric_key,
                self.k_index,
            )
            hit = self.cache.get(key)
            if hit is not None:
                cached[idx] = hit
            else:
                miss_indices.append(idx)
                miss_items.append(item)

        if miss_items:
            fresh = self.inner.grade(conversation, miss_items)
            for miss_idx, item, verdict in zip(miss_indices, miss_items, fresh, strict=True):
                rubric_key = item.criterion_id or (
                    f"{item.criterion}|points={item.points}|tags={tuple(item.tags)}"
                )
                key = self.cache.make_key(
                    self.model_fingerprint,
                    self.prompt_sha,
                    conversation,
                    rubric_key,
                    self.k_index,
                )
                self.cache.put(key, verdict)
                cached[miss_idx] = verdict

        return [cached[i] for i in range(len(rubric_items))]
