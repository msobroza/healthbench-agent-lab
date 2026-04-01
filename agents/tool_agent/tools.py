"""Medical reference tools registry for agent pipelines.

Imports all tool modules to trigger their ``@register_tool`` registration.
After importing this module, all tools are available via the tool registry::

    from healthbench_agent.agent import get_tools
    tools = get_tools(["drug_reference", "symptom_checker", "emergency_flag"])

Individual tool functions are re-exported here for convenience.
"""

from .drug_reference import drug_reference
from .emergency_flag import EMERGENCY_KEYWORDS, URGENT_KEYWORDS, emergency_flag
from .symptom_checker import symptom_checker

__all__ = [
    "drug_reference",
    "symptom_checker",
    "emergency_flag",
    "EMERGENCY_KEYWORDS",
    "URGENT_KEYWORDS",
]
