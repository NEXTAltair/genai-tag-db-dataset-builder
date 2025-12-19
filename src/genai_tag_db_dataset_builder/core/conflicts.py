"""衝突検出結果の出力（レポート）.

タグDBマージ時の衝突（type_id不一致、alias変更など）をCSVとして出力します。
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def export_conflict_reports(
    conflicts: dict[str, pl.DataFrame],
    output_dir: Path | str,
) -> dict[str, Path | None]:
    """衝突レポートをCSVファイルとして出力する.

    Args:
        conflicts: detect_conflicts() の戻り値（衝突情報）
        output_dir: 出力ディレクトリ

    Returns:
        出力したCSVのパス（衝突が無ければ None）
        - "type_conflicts": type_id_conflicts.csv
        - "alias_changes": alias_changes.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[str, Path | None] = {}

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
