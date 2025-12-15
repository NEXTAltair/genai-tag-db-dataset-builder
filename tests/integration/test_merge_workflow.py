"""Integration tests for merge workflow.

This module tests the complete merge workflow including:
- Tag normalization
- Merge with existing tags
- Deprecated tags processing
- Conflict detection
"""

import polars as pl
import pytest

from genai_tag_db_dataset_builder.core.merge import detect_conflicts, merge_tags, process_deprecated_tags


@pytest.mark.integration
class TestMergeWorkflow:
    """マージワークフロー統合テスト."""

    def test_complete_merge_workflow(self) -> None:
        """完全なマージワークフローのテスト."""
        # 1. 既存タグセット
        existing_tags = {"witch", "mage", "spiked collar"}

        # 2. 新規データ（source_tag列あり）
        new_df = pl.DataFrame(
            {
                "source_tag": [
                    "Witch",  # 既存（大文字）
                    "new_tag",  # 新規
                    "Another_Tag",  # 新規
                    "spiked_collar",  # 既存
                ]
            }
        )

        # 3. マージ実行
        next_tag_id = 100
        result = merge_tags(existing_tags, new_df, next_tag_id)

        # 4. 結果確認
        assert len(result) == 2  # new_tag, another tag のみ
        assert result["tag_id"].to_list() == [100, 101]
        assert "another tag" in result["tag"].to_list()
        assert "new tag" in result["tag"].to_list()

    def test_deprecated_tags_workflow(self) -> None:
        """deprecated_tags処理ワークフローのテスト."""
        # 1. tags_mapping作成（tag → tag_id）
        tags_mapping = {"witch": 1, "mage": 2, "wizard": 3}

        # 2. deprecated_tags処理
        records = process_deprecated_tags("witch", "mage,wizard", 1, tags_mapping)

        # 3. 結果確認
        assert len(records) == 3

        # canonical
        assert records[0]["tag_id"] == 1
        assert records[0]["alias"] == 0
        assert records[0]["preferred_tag_id"] == 1

        # alias1
        assert records[1]["tag_id"] == 2
        assert records[1]["alias"] == 1
        assert records[1]["preferred_tag_id"] == 1

        # alias2
        assert records[2]["tag_id"] == 3
        assert records[2]["alias"] == 1
        assert records[2]["preferred_tag_id"] == 1

    def test_conflict_detection_workflow(self) -> None:
        """衝突検出ワークフローのテスト."""
        # 1. 既存TAG_STATUS（tag列JOIN済み）
        existing_df = pl.DataFrame(
            {
                "tag": ["witch", "mage"],
                "format_id": [1, 1],
                "type_id": [4, 2],
                "alias": [0, 0],
            }
        )

        # 2. 新規データ（tag列あり）
        new_df = pl.DataFrame(
            {
                "tag": ["witch", "mage"],
                "format_id": [1, 1],
                "type_id": [0, 2],  # witchのtype_idが4→0に変更
                "alias": [1, 0],  # witchのaliasが0→1に変更
            }
        )

        # 3. 衝突検出
        conflicts = detect_conflicts(existing_df, new_df)

        # 4. 結果確認
        assert len(conflicts["type_conflicts"]) == 1
        assert conflicts["type_conflicts"]["tag"][0] == "witch"

        assert len(conflicts["alias_changes"]) == 1
        assert conflicts["alias_changes"]["tag"][0] == "witch"

    def test_full_workflow_integration(self) -> None:
        """フルワークフロー統合テスト."""
        # 1. 既存タグセット
        existing_tags = {"witch", "mage"}

        # 2. 新規データ
        new_df = pl.DataFrame({"source_tag": ["Wizard", "Sorcerer", "Witch"]})

        # 3. マージ
        next_tag_id = 100
        merged = merge_tags(existing_tags, new_df, next_tag_id)

        # 4. マージ結果確認
        assert len(merged) == 2  # wizard, sorcerer のみ
        assert "wizard" in merged["tag"].to_list()
        assert "sorcerer" in merged["tag"].to_list()

        # 5. tags_mapping更新
        tags_mapping = {"witch": 1, "mage": 2, "wizard": 100, "sorcerer": 101}

        # 6. deprecated_tags処理
        records = process_deprecated_tags("witch", "mage", 1, tags_mapping)

        # 7. TAG_STATUSレコード確認
        assert len(records) == 2  # canonical + alias1
        assert records[0]["tag_id"] == 1  # witch (canonical)
        assert records[1]["tag_id"] == 2  # mage (alias)

        # 8. 衝突検出（既存なし→衝突なし）
        existing_df = pl.DataFrame(
            {"tag": [], "format_id": [], "type_id": [], "alias": []},
            schema={"tag": pl.String, "format_id": pl.Int64, "type_id": pl.Int64, "alias": pl.Int64},
        )
        new_df = pl.DataFrame({"tag": ["witch"], "format_id": [1], "type_id": [4], "alias": [0]})

        conflicts = detect_conflicts(existing_df, new_df)

        assert len(conflicts["type_conflicts"]) == 0
        assert len(conflicts["alias_changes"]) == 0
