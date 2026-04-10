"""File-based cache for individual judge verdicts.

Enables iterating on metric definitions, filters, or output formats without
re-paying the LLM. Cache key is

    sha256(judge_model || judge_prompt_sha || conversation_hash || k_index || rubric_text)

where conversation_hash is sha256 of the JSON-serialised MessageList.

Exposes :class:`VerdictCache` (the on-disk store) and
:func:`make_verdict_cache_key` (the pure key-construction function). The
:class:`CachedJudgeGrader` proxy that wraps the cache lives in
``cache/cached_judge.py``.

Thread safety
-------------
``VerdictCache`` is safe to share across worker threads under the
following contract:

* Hit/miss counter mutations are guarded by an internal ``threading.Lock``
  so :meth:`VerdictCache.stats` returns consistent values when called from
  any thread.
* On-disk reads in :meth:`VerdictCache.get` are unlocked: small JSON
  files are read with a single ``Path.read_text`` call and the OS-level
  read is already atomic.
* On-disk writes in :meth:`VerdictCache.put` use a temp-file +
  ``os.replace`` dance, which is atomic on POSIX and Windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import asdict
from pathlib import Path

from healthbench_agent.domain.conversation import MessageList
from healthbench_agent.domain.evaluation import CriterionVerdict

logger = logging.getLogger(__name__)


def make_verdict_cache_key(
    judge_model: str,
    judge_prompt_sha: str,
    conversation: MessageList,
    rubric_text: str,
    k_index: int,
) -> str:
    """Compute the deterministic sha256 cache key for a verdict.

    Pure function — no I/O, no side effects. Callers can use this
    directly (without a :class:`VerdictCache` instance) when they only
    need to derive a key.

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
    conv_hash = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode("utf-8")).hexdigest()
    payload = "||".join([judge_model, judge_prompt_sha, conv_hash, str(k_index), rubric_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        self._lock = threading.Lock()

    def make_key(
        self,
        judge_model: str,
        judge_prompt_sha: str,
        conversation: MessageList,
        rubric_text: str,
        k_index: int,
    ) -> str:
        """Compute the deterministic sha256 cache key.

        Thin backwards-compatible delegate to
        :func:`make_verdict_cache_key`. Prefer the free function in new
        code when you only need a key and not the cache instance.

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
        return make_verdict_cache_key(
            judge_model=judge_model,
            judge_prompt_sha=judge_prompt_sha,
            conversation=conversation,
            rubric_text=rubric_text,
            k_index=k_index,
        )

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
            with self._lock:
                self._misses += 1
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Corrupt cache entry at %s — treating as miss", path)
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
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
        with self._lock:
            hits = self._hits
            misses = self._misses
        return {"hits": hits, "misses": misses, "size_bytes": size}
