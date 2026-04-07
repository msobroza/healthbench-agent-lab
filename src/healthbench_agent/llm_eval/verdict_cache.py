"""File-based cache for individual judge verdicts.

Enables iterating on metric definitions, filters, or output formats without
re-paying the LLM. Cache key is

    sha256(judge_model || judge_prompt_sha || conversation_hash || k_index || rubric_text)

where conversation_hash is sha256 of the JSON-serialised MessageList.

Includes ``CachedJudgeGrader``, a thin proxy that wraps any
``JudgeGrader`` and consults the cache before delegating, so the
``JudgeGrader`` ABC stays unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any  # noqa: F401

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict
from healthbench_agent.domain.judge import JudgeGrader
from healthbench_agent.domain.rubric import RubricItem

logger = logging.getLogger(__name__)


def _default_cache_root() -> Path:
    """Return the default root directory for cached verdicts.

    Respects ``XDG_CACHE_HOME`` and falls back to ``~/.cache``.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "healthbench_agent" / "verdicts"


class VerdictCache:
    """File-based cache for individual judge verdicts.

    Each cache entry is one tiny JSON file under ``root/first2/rest.json``,
    like git's loose object format. Sharding by the first two hex characters
    of the key keeps any single directory from growing unbounded.

    Attributes:
        root: Directory under which cache entries are stored.
        enabled: When False, ``get`` always returns None and ``put`` is a
            no-op — useful for disabling the cache without rewriting call
            sites.
    """

    def __init__(
        self,
        root: Path | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialise the cache.

        Args:
            root: Directory where cache files live. Defaults to
                ``$XDG_CACHE_HOME/healthbench_agent/verdicts``.
            enabled: Toggle that disables reads and writes when False.
        """
        self.root = root or _default_cache_root()
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        judge_model: str,
        judge_prompt_sha: str,
        conversation: MessageList,
        rubric_text: str,
        k_index: int,
    ) -> str:
        """Compute the deterministic sha256 cache key.

        Args:
            judge_model: Identifier of the judge model (e.g. ``"gpt-4o"``).
            judge_prompt_sha: Hash of the rendered judge prompt template.
            conversation: Full conversation being graded.
            rubric_text: Text of the rubric criterion being graded.
            k_index: Index of this sample within a multi-sample majority
                vote, so independent samples get distinct keys.

        Returns:
            Hex sha256 digest uniquely identifying the (model, prompt,
            conversation, rubric, k) tuple.
        """
        conv_hash = hashlib.sha256(
            json.dumps(conversation, sort_keys=True).encode("utf-8")
        ).hexdigest()
        payload = "||".join([judge_model, judge_prompt_sha, conv_hash, str(k_index), rubric_text])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        """Return the on-disk path for a given cache key."""
        return self.root / key[:2] / f"{key[2:]}.json"

    def get(self, key: str) -> CriterionVerdict | None:
        """Return the cached verdict for ``key``, or None on miss.

        Args:
            key: Cache key produced by :meth:`make_key`.

        Returns:
            The stored :class:`CriterionVerdict` on hit, otherwise None.
            Also returns None when the cache is disabled or the file is
            unreadable/corrupt.
        """
        if not self.enabled:
            return None
        path = self._path_for(key)
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Corrupt cache entry at %s — treating as miss", path)
            self._misses += 1
            return None
        self._hits += 1
        return CriterionVerdict(
            criterion=data["criterion"],
            criteria_met=data["criteria_met"],
            explanation=data.get("explanation", ""),
            confidence=data.get("confidence", 1.0),
        )

    def put(self, key: str, verdict: CriterionVerdict) -> None:
        """Write ``verdict`` to the cache under ``key``.

        Args:
            key: Cache key produced by :meth:`make_key`.
            verdict: The verdict to persist.
        """
        if not self.enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(verdict)))
        os.replace(tmp, path)

    def clear(self) -> None:
        """Delete every cached verdict and remove the root directory."""
        shutil.rmtree(self.root, ignore_errors=True)

    def stats(self) -> dict[str, int]:
        """Return runtime statistics for this cache instance.

        Returns:
            Dict with keys ``hits`` (session hits), ``misses`` (session
            misses), and ``size_bytes`` (total on-disk size of cache files).
        """
        size = 0
        if self.root.exists():
            for shard in self.root.iterdir():
                if shard.is_dir():
                    for entry in shard.iterdir():
                        size += entry.stat().st_size
        return {"hits": self._hits, "misses": self._misses, "size_bytes": size}


class CachedJudgeGrader(JudgeGrader):
    """JudgeGrader proxy that consults a VerdictCache before delegating.

    Wraps any JudgeGrader and intercepts ``grade()`` to deduplicate
    verdicts. The cache key components that are constant across one
    k-pass (model fingerprint, prompt sha, k_index) are baked in at
    construction time so the proxy can compute the full key from just
    the (conversation, rubric_text) tuple — no introspection of the
    inner grader required.
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
            key = self.cache.make_key(
                self.model_fingerprint,
                self.prompt_sha,
                conversation,
                item.criterion,
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
                key = self.cache.make_key(
                    self.model_fingerprint,
                    self.prompt_sha,
                    conversation,
                    item.criterion,
                    self.k_index,
                )
                self.cache.put(key, verdict)
                cached[miss_idx] = verdict

        return [cached[i] for i in range(len(rubric_items))]
