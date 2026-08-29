"""Standalone veRL reward entry point.

veRL loads custom reward files by path, so this module uses absolute imports
instead of package-relative imports.
"""

from reasonsql.rewards.bird_reward import compute_score

__all__ = ["compute_score"]

