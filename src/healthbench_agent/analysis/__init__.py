"""HealthBench analysis subpackage — registry, utilities, and exploration analyses.

Public API:
    from healthbench_agent.analysis import register_analysis, run_one, run_category, run_all
    from healthbench_agent.analysis import AnalysisEntry
    from healthbench_agent.analysis import DEFAULT_PERCENTILES, series_stats, save_csv
"""

from .registry import AnalysisEntry, register_analysis, run_all, run_category, run_one
from .utils import DEFAULT_PERCENTILES, parse_tag_prefixes, save_csv, series_stats

__all__ = [
    # registry
    "AnalysisEntry",
    "register_analysis",
    "run_one",
    "run_category",
    "run_all",
    # utils
    "DEFAULT_PERCENTILES",
    "series_stats",
    "save_csv",
    "parse_tag_prefixes",
]
