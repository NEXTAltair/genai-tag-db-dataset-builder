"""Unit tests for merge functionality."""

import polars as pl

from genai_tag_db_dataset_builder.core.merge import (
    detect_conflicts,
    merge_tags,
    process_deprecated_tags,
)


class TestMergeTags:
    """merge_tags関数のテスト."""

    def test_basic_merge(self) -> None:
        """基本的なマージテスト."""
        existing_tags = {"witch", "spiked collar"}
        new_df = pl.DataFrame({"source_tag": ["new_tag", "another_tag"]})

        result = merge_tags(existing_tags, new_df, 100)

        assert len(result) == 2
        assert "tag_id" in result.columns
        assert "tag" in result.columns
        assert "source_tag" in result.columns

        # tag_idがソート順に採番されていることを確認
        assert result["tag_id"].to_list() == [100, 101]

    def test_merge_with_duplicates(self) -> None:
        """重複排除のテスト."""
        existing_tags = {"witch"}
        new_df = pl.DataFrame({"source_tag": ["Witch", "new_tag", "New_Tag"]})

        result = merge_tags(existing_tags, new_df, 100)

        # "Witch"は既存、"new_tag"と"New_Tag"は正規化後同じなので1件のみ
        assert len(result) == 1
        assert result["tag"][0] == "new tag"

    def test_merge_with_existing_tags(self) -> None:
        """既存タグが除外されることのテスト."""
        existing_tags = {"witch", "new tag"}
        new_df = pl.DataFrame({"source_tag": ["Witch", "new_tag", "unique_tag"]})

        result = merge_tags(existing_tags, new_df, 100)

        # "Witch"と"new_tag"は既存、"unique_tag"のみ追加
        assert len(result) == 1
        assert result["tag"][0] == "unique tag"

    def test_merge_empty_new_df(self) -> None:
        """新規DataFrameが空の場合のテスト."""
        existing_tags = {"witch"}
        new_df = pl.DataFrame({"source_tag": []})

        result = merge_tags(existing_tags, new_df, 100)

        assert len(result) == 0

    def test_merge_tags_raises_on_missing_source_tag_column(self) -> None:
        """source_tag列がない場合にValueErrorを発生させることを確認."""
        import pytest

        existing_tags = {"witch"}
        # source_tag列のないDataFrame
        new_df = pl.DataFrame(
            {
                "tag": ["Witch", "new_tag"],
                "type_id": [4, 0],
            }
        )

        with pytest.raises(ValueError, match="merge_tags\\(\\) requires 'source_tag' column"):
            merge_tags(existing_tags, new_df, 100)


class TestProcessDeprecatedTags:
    """process_deprecated_tags関数のテスト."""

    def test_with_deprecated_tags(self) -> None:
        """deprecated_tags処理のテスト."""
        tags_mapping = {"witch": 1, "mage": 2, "wizard": 3}
        records = process_deprecated_tags("witch", "mage,wizard", 1, tags_mapping)

        # canonical + 2 aliases = 3レコード
        assert len(records) == 3

        # canonical（alias=0）
        assert records[0]["tag_id"] == 1
        assert records[0]["alias"] == 0
        assert records[0]["preferred_tag_id"] == 1

        # alias1（alias=1）
        assert records[1]["tag_id"] == 2
        assert records[1]["alias"] == 1
        assert records[1]["preferred_tag_id"] == 1

        # alias2（alias=1）
        assert records[2]["tag_id"] == 3
        assert records[2]["alias"] == 1
        assert records[2]["preferred_tag_id"] == 1

    def test_without_deprecated_tags(self) -> None:
        """deprecated_tags無しの場合のテスト."""
        tags_mapping = {"witch": 1}
        records = process_deprecated_tags("witch", "", 1, tags_mapping)

        # canonicalのみ
        assert len(records) == 1
        assert records[0]["alias"] == 0

    def test_with_nonexistent_alias(self) -> None:
        """存在しないaliasの場合のテスト."""
        tags_mapping = {"witch": 1}
        records = process_deprecated_tags("witch", "nonexistent", 1, tags_mapping)

        # canonicalのみ（nonexistentはtags_mappingに無いのでスキップ）
        assert len(records) == 1

    def test_process_deprecated_tags_logs_missing_aliases(self) -> None:
        """存在しないaliasがある場合にWARNINGログを出力することを確認."""
        from unittest.mock import patch

        tags_mapping = {"witch": 1, "mage": 2}
        # "nonexistent1"と"nonexistent2"はmappingに存在しない
        deprecated_tags = "mage,nonexistent1,nonexistent2"

        with patch("genai_tag_db_dataset_builder.core.merge.logger.warning") as mock_warning:
            records = process_deprecated_tags("witch", deprecated_tags, 1, tags_mapping)

        # canonicalとmageの2レコードのみ（nonexistent1/2はスキップ）
        assert len(records) == 2

        # logger.warningが1回呼ばれたことを確認
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]

        # ログメッセージの内容を確認
        assert "Skipped 2 alias(es) missing from TAGS table" in warning_message
        assert "nonexistent1" in warning_message
        assert "nonexistent2" in warning_message
        assert "canonical: witch" in warning_message
        assert "format_id: 1" in warning_message
        assert "Ensure two-pass processing" in warning_message


class TestDetectConflicts:
    """detect_conflicts関数のテスト."""

    def test_type_id_conflict(self) -> None:
        """type_id不一致検出のテスト."""
        existing = pl.DataFrame(
            {
                "tag": ["witch", "mage"],
                "format_id": [1, 1],
                "type_id": [4, 2],
                "alias": [0, 0],
            }
        )
        new = pl.DataFrame(
            {
                "tag": ["witch", "other"],
                "format_id": [1, 1],
                "type_id": [0, 2],
                "alias": [0, 0],
            }
        )
        conflicts = detect_conflicts(existing, new)

        # "witch"のtype_idが4→0に変更
        assert len(conflicts["type_conflicts"]) == 1
        assert conflicts["type_conflicts"]["tag"][0] == "witch"

    def test_alias_change(self) -> None:
        """alias変更検出のテスト."""
        existing = pl.DataFrame(
            {
                "tag": ["witch"],
                "format_id": [1],
                "type_id": [4],
                "alias": [0],
            }
        )
        new = pl.DataFrame(
            {
                "tag": ["witch"],
                "format_id": [1],
                "type_id": [4],
                "alias": [1],
            }
        )
        conflicts = detect_conflicts(existing, new)

        # "witch"がcanonical→aliasに変更
        assert len(conflicts["alias_changes"]) == 1
        assert conflicts["alias_changes"]["tag"][0] == "witch"

    def test_no_conflicts(self) -> None:
        """衝突無しの場合のテスト."""
        existing = pl.DataFrame(
            {
                "tag": ["witch"],
                "format_id": [1],
                "type_id": [4],
                "alias": [0],
            }
        )
        new = pl.DataFrame(
            {
                "tag": ["other"],
                "format_id": [1],
                "type_id": [2],
                "alias": [0],
            }
        )
        conflicts = detect_conflicts(existing, new)

        assert len(conflicts["type_conflicts"]) == 0
        assert len(conflicts["alias_changes"]) == 0
