"""HealthBench analysis subpackage — registry, utilities, and analyses.

Public API:
    from healthbench_agent.analysis import register_analysis, run_one, run_category, run_all
    from healthbench_agent.analysis import AnalysisEntry
    from healthbench_agent.analysis import DEFAULT_PERCENTILES, series_stats, save_csv
"""

# Register analyses by importing their modules.
from . import exploration as exploration  # noqa: F401
from . import insights as insights  # noqa: F401
from . import visualization as visualization  # noqa: F401
from .registry import AnalysisEntry, register_analysis, run_all, run_category, run_one
from .utils import (
    DEFAULT_PERCENTILES,
    build_rubric_dataframe,
    build_sample_dataframe,
    parse_tag_prefixes,
    save_csv,
    series_stats,
    tag_value,
    total_penalty_points,
    total_positive_points,
    total_prompt_chars,
)

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
    "tag_value",
    "total_positive_points",
    "total_penalty_points",
    "total_prompt_chars",
    "build_rubric_dataframe",
    "build_sample_dataframe",
]
