"""Conflict detection and reporting.

This module provides conflict detection and CSV report generation for
tag database merging.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def export_conflict_reports(
    conflicts: dict[str, pl.DataFrame],
    output_dir: Path | str,
) -> dict[str, Path]:
    """衝突レポートをCSVファイルに出力.

    Args:
        conflicts: detect_conflicts()の返却値
        output_dir: 出力ディレクトリ

    Returns:
        生成されたCSVファイルのパス辞書
        - "type_conflicts": type_id_conflicts.csvのパス
        - "alias_changes": alias_changes.csvのパス

    Examples:
        >>> conflicts = detect_conflicts(existing_df, new_df)
        >>> paths = export_conflict_reports(conflicts, "reports/")
        >>> paths["type_conflicts"]
        PosixPath('reports/type_id_conflicts.csv')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths = {}

    # type_id不一致レポート
    type_conflicts_path = output_dir / "type_id_conflicts.csv"
    if len(conflicts["type_conflicts"]) > 0:
        conflicts["type_conflicts"].write_csv(type_conflicts_path)
        result_paths["type_conflicts"] = type_conflicts_path
    else:
        result_paths["type_conflicts"] = None

    # alias変更レポート
    alias_changes_path = output_dir / "alias_changes.csv"
    if len(conflicts["alias_changes"]) > 0:
        conflicts["alias_changes"].write_csv(alias_changes_path)
        result_paths["alias_changes"] = alias_changes_path
    else:
        result_paths["alias_changes"] = None

    return result_paths
