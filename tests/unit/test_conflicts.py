"""Unit tests for conflict reporting."""

from pathlib import Path

import polars as pl

from genai_tag_db_dataset_builder.core.conflicts import export_conflict_reports


class TestExportConflictReports:
    """export_conflict_reports関数のテスト."""

    def test_export_with_conflicts(self, tmp_path: Path) -> None:
        """衝突ありの場合のCSV出力テスト."""
        # 衝突データ作成
        type_conflicts = pl.DataFrame(
            {
                "tag": ["witch", "mage"],
                "format_id": [1, 1],
                "type_id": [4, 2],
                "type_id_new": [0, 3],
            }
        )

        alias_changes = pl.DataFrame({"tag": ["wizard"], "format_id": [1], "alias": [0], "alias_new": [1]})

        conflicts = {"type_conflicts": type_conflicts, "alias_changes": alias_changes}

        # CSV出力
        output_dir = tmp_path / "reports"
        paths = export_conflict_reports(conflicts, output_dir)

        # 出力確認
        assert paths["type_conflicts"] is not None
        assert paths["type_conflicts"].exists()
        assert paths["alias_changes"] is not None
        assert paths["alias_changes"].exists()

        # CSVファイル読み込み確認
        type_df = pl.read_csv(paths["type_conflicts"])
        assert len(type_df) == 2
        assert "tag" in type_df.columns

        alias_df = pl.read_csv(paths["alias_changes"])
        assert len(alias_df) == 1
        assert "tag" in alias_df.columns

    def test_export_without_conflicts(self, tmp_path: Path) -> None:
        """衝突なしの場合のCSV出力テスト."""
        # 空の衝突データ
        type_conflicts = pl.DataFrame(
            {"tag": [], "format_id": [], "type_id": [], "type_id_new": []},
            schema={"tag": pl.String, "format_id": pl.Int64, "type_id": pl.Int64, "type_id_new": pl.Int64},
        )

        alias_changes = pl.DataFrame(
            {"tag": [], "format_id": [], "alias": [], "alias_new": []},
            schema={"tag": pl.String, "format_id": pl.Int64, "alias": pl.Int64, "alias_new": pl.Int64},
        )

        conflicts = {"type_conflicts": type_conflicts, "alias_changes": alias_changes}

        # CSV出力
        output_dir = tmp_path / "reports"
        paths = export_conflict_reports(conflicts, output_dir)

        # 出力確認（衝突なしの場合はNone）
        assert paths["type_conflicts"] is None
        assert paths["alias_changes"] is None

    def test_output_directory_creation(self, tmp_path: Path) -> None:
        """出力ディレクトリ自動作成のテスト."""
        # 存在しないディレクトリを指定
        output_dir = tmp_path / "new" / "reports"
        assert not output_dir.exists()

        # 衝突データ作成
        type_conflicts = pl.DataFrame(
            {"tag": ["witch"], "format_id": [1], "type_id": [4], "type_id_new": [0]}
        )

        alias_changes = pl.DataFrame({"tag": [], "format_id": [], "alias": [], "alias_new": []})

        conflicts = {"type_conflicts": type_conflicts, "alias_changes": alias_changes}

        # CSV出力
        paths = export_conflict_reports(conflicts, output_dir)

        # ディレクトリが作成されたことを確認
        assert output_dir.exists()
        assert paths["type_conflicts"].exists()
