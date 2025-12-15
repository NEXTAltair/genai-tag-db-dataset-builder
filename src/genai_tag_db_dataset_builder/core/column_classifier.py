"""入力カラム（tag/source_tag）の意味推定.

CSV/JSON/Parquet等の表形式入力において、`tag` 列が
- 生タグ（source_tag。例: `witch_hat`）なのか
- 正規化済み（TAGS.tag 相当。例: `witch hat`）なのか
を推定するためのモジュール。

推定が曖昧な場合は UNKNOWN を返し、ビルドは継続しつつレポートで人力修正へ誘導する。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import polars as pl

from genai_tag_db_dataset_builder.core.normalize import normalize_tag


class TagColumnType(str, Enum):
    """tag列の意味の推定結果."""

    NORMALIZED = "normalized"  # 既に正規化済み（TAGS.tag 相当）
    SOURCE = "source"  # 生タグ（source_tag 相当）
    UNKNOWN = "unknown"  # 自動判定不可（人力修正）


@dataclass(frozen=True)
class TagColumnSignals:
    underscore_ratio: float
    escaped_paren_ratio: float
    normalize_change_ratio: float
    decision: TagColumnType
    confidence: str
    source_signals: int
    normalized_signals: int

    def as_dict(self) -> dict[str, object]:
        return {
            "underscore_ratio": self.underscore_ratio,
            "escaped_paren_ratio": self.escaped_paren_ratio,
            "normalize_change_ratio": self.normalize_change_ratio,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "source_signals": self.source_signals,
            "normalized_signals": self.normalized_signals,
        }


def _coerce_str_list(values: list[object]) -> list[str]:
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def calculate_underscore_ratio(tags: list[str]) -> float:
    """タグ集合におけるアンダースコア含有率."""
    if not tags:
        return 0.0
    return sum(1 for tag in tags if "_" in tag) / len(tags)


def detect_escaped_parentheses(tags: list[str]) -> float:
    r"""タグ集合における括弧エスケープ（\(\)）含有率."""
    if not tags:
        return 0.0
    return sum(1 for tag in tags if r"\(" in tag or r"\)" in tag) / len(tags)


def calculate_normalize_change_ratio(tags: list[str]) -> float:
    """normalize_tag() 適用で変化する割合."""
    if not tags:
        return 0.0
    return sum(1 for tag in tags if normalize_tag(tag) != tag) / len(tags)


def classify_tag_column(
    df: pl.DataFrame,
    column_name: str = "tag",
    thresholds: dict[str, float] | None = None,
) -> tuple[TagColumnType, dict[str, object]]:
    """tag列が SOURCE / NORMALIZED / UNKNOWN のどれかを推定する.

シグナル（ざっくり）:
- underscore_ratio が高い: SOURCE 寄り
- escaped_paren_ratio が高い: NORMALIZED 寄り（DBツールの括弧エスケープ痕跡）
- normalize_change_ratio が低い: NORMALIZED 寄り（既に正規化済み）

Returns:
    (TagColumnType, signals_dict)
"""
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    raw_values = df[column_name].to_list()
    tags = _coerce_str_list(raw_values)
    if not tags:
        raise ValueError(f"Column '{column_name}' is empty")

    default_thresholds: dict[str, float] = {
        "underscore_threshold": 0.7,  # SOURCE判定の強シグナル
        "escaped_paren_threshold": 0.3,  # NORMALIZED判定の強シグナル
        "normalize_change_threshold": 0.5,  # SOURCE判定の強シグナル
        "unknown_margin": 0.1,  # 低い/高いの“端”を判定する幅
    }
    if thresholds is None:
        thresholds = default_thresholds
    else:
        thresholds = {**default_thresholds, **thresholds}

    underscore_ratio = calculate_underscore_ratio(tags)
    escaped_paren_ratio = detect_escaped_parentheses(tags)
    normalize_change_ratio = calculate_normalize_change_ratio(tags)

    source_signals = 0
    normalized_signals = 0

    # Signal 1: underscore
    if underscore_ratio >= thresholds["underscore_threshold"]:
        source_signals += 1
    elif underscore_ratio <= thresholds["unknown_margin"]:
        normalized_signals += 1

    # Signal 2: escaped parentheses -> NORMALIZED
    if escaped_paren_ratio >= thresholds["escaped_paren_threshold"]:
        normalized_signals += 1

    # Signal 3: normalize_tag() changes
    if normalize_change_ratio >= thresholds["normalize_change_threshold"]:
        source_signals += 1
    elif normalize_change_ratio <= thresholds["unknown_margin"]:
        normalized_signals += 1

    if source_signals >= 2:
        decision = TagColumnType.SOURCE
        confidence = "high" if source_signals == 3 else "medium"
    elif normalized_signals >= 2:
        decision = TagColumnType.NORMALIZED
        confidence = "high" if normalized_signals == 3 else "medium"
    else:
        decision = TagColumnType.UNKNOWN
        confidence = "low"

    signals = TagColumnSignals(
        underscore_ratio=underscore_ratio,
        escaped_paren_ratio=escaped_paren_ratio,
        normalize_change_ratio=normalize_change_ratio,
        decision=decision,
        confidence=confidence,
        source_signals=source_signals,
        normalized_signals=normalized_signals,
    )
    return decision, signals.as_dict()

