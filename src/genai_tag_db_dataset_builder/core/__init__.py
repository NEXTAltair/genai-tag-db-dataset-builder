"""タグDB構築のコア処理群.

- 正規化（入力タグ → TAGS.tag）
- マージ（既存との差分抽出、採番）
- 衝突検出（type_id/alias 等）
"""

from .merge import detect_conflicts, merge_tags, process_deprecated_tags
from .normalize import normalize_tag

__all__ = [
    "normalize_tag",
    "merge_tags",
    "process_deprecated_tags",
    "detect_conflicts",
]
