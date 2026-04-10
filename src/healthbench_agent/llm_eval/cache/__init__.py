"""On-disk verdict cache and proxy ``JudgeGrader`` that consults it.

Public surface:

* :class:`VerdictCache` — thread-safe on-disk store.
* :func:`make_verdict_cache_key` — pure cache-key constructor.
* :class:`CachedJudgeGrader` — JudgeGrader proxy wrapping a VerdictCache.
"""

from .cached_judge import CachedJudgeGrader
from .store import VerdictCache, make_verdict_cache_key

__all__ = [
    "CachedJudgeGrader",
    "VerdictCache",
    "make_verdict_cache_key",
]
