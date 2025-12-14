"""Core functionality for tag database building."""

from .merge import detect_conflicts, merge_tags, process_deprecated_tags
from .normalize import normalize_tag

__all__ = [
    "normalize_tag",
    "merge_tags",
    "process_deprecated_tags",
    "detect_conflicts",
]
