"""Tool-augmented agent package — Architecture B.

Importing this package triggers @register_tool registration for all
medical reference tools, making them available via the tool registry.
"""

# Import tools package to trigger @register_tool registration.
import tools  # noqa: F401
