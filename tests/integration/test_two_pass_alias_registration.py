"""Integration test for two-pass alias registration workflow.

This test verifies that the two-pass processing in builder.py correctly prevents
data loss by ensuring all tags (including those in deprecated_tags) are registered
in TAGS table before TAG_STATUS records are created.
"""

import polars as pl

from genai_tag_db_dataset_builder.builder import _extract_all_tags_from_deprecated
from genai_tag_db_dataset_builder.core.merge import merge_tags, process_deprecated_tags


class TestTwoPassAliasRegistration:
    """Two-pass処理による完全なalias登録のテスト."""

    def test_extract_all_tags_from_deprecated(self) -> None:
        """_extract_all_tags_from_deprecated()がsource_tagとdeprecated_tagsの全タグを抽出."""
        df = pl.DataFrame(
            {
                "source_tag": ["witch", "mage", "warrior"],
                "deprecated_tags": ["sorceress,enchantress", "wizard,warlock", ""],
            }
        )

        all_tags = _extract_all_tags_from_deprecated(df)

        # source_tag (3) + deprecated_tags (4) = 7 unique tags
        expected_tags = {"witch", "mage", "warrior", "sorceress", "enchantress", "wizard", "warlock"}
        assert all_tags == expected_tags

    def test_extract_handles_missing_columns(self) -> None:
        """deprecated_tags列が欠損している場合の処理."""
        df = pl.DataFrame({"source_tag": ["witch", "mage"]})

        all_tags = _extract_all_tags_from_deprecated(df)

        # source_tagのみ抽出
        assert all_tags == {"witch", "mage"}

    def test_extract_handles_none_values(self) -> None:
        """deprecated_tags列にNone値がある場合の処理."""
        df = pl.DataFrame(
            {
                "source_tag": ["witch", "mage"],
                "deprecated_tags": ["sorceress", None],
            }
        )

        all_tags = _extract_all_tags_from_deprecated(df)

        # None値はスキップ
        assert all_tags == {"witch", "mage", "sorceress"}

    def test_two_pass_prevents_data_loss(self) -> None:
        """2パス処理により、aliasが欠損せずに全て登録されることを確認."""
        from unittest.mock import patch

        # Simulate CSV data with deprecated_tags
        csv_df = pl.DataFrame(
            {
                "source_tag": ["witch", "mage"],
                "deprecated_tags": ["sorceress,enchantress", "wizard"],
            }
        )

        existing_tags: set[str] = set()

        # === Pass 1: Extract all tags ===
        all_tags = _extract_all_tags_from_deprecated(csv_df)

        # Verify all tags are collected
        assert all_tags == {"witch", "mage", "sorceress", "enchantress", "wizard"}

        # Merge all tags into TAGS table
        all_tags_df = pl.DataFrame({"source_tag": sorted(all_tags)})
        merged_df = merge_tags(existing_tags, all_tags_df, next_tag_id=1)

        # Build tags_mapping
        tags_mapping = {row["tag"]: row["tag_id"] for row in merged_df.to_dicts()}

        # Verify all tags have tag_id
        assert len(tags_mapping) == 5
        assert "witch" in tags_mapping
        assert "sorceress" in tags_mapping
        assert "wizard" in tags_mapping

        # === Pass 2: Create TAG_STATUS records ===
        with patch("genai_tag_db_dataset_builder.core.merge.logger.warning") as mock_warning:
            # Process first row: witch -> sorceress, enchantress
            records_1 = process_deprecated_tags(
                canonical_tag="witch",
                deprecated_tags="sorceress,enchantress",
                format_id=1,
                tags_mapping=tags_mapping,
            )

            # Process second row: mage -> wizard
            records_2 = process_deprecated_tags(
                canonical_tag="mage",
                deprecated_tags="wizard",
                format_id=1,
                tags_mapping=tags_mapping,
            )

        # No warnings should be logged (all aliases found in tags_mapping)
        mock_warning.assert_not_called()

        # Verify TAG_STATUS records created correctly
        assert len(records_1) == 3  # witch (canonical) + sorceress + enchantress
        assert len(records_2) == 2  # mage (canonical) + wizard

        # Verify alias records have correct preferred_tag_id
        witch_id = tags_mapping["witch"]
        mage_id = tags_mapping["mage"]

        # records_1: sorceress and enchantress should point to witch
        alias_records_1 = [r for r in records_1 if r["alias"] == 1]
        assert len(alias_records_1) == 2
        assert all(r["preferred_tag_id"] == witch_id for r in alias_records_1)

        # records_2: wizard should point to mage
        alias_records_2 = [r for r in records_2 if r["alias"] == 1]
        assert len(alias_records_2) == 1
        assert alias_records_2[0]["preferred_tag_id"] == mage_id

    def test_single_pass_causes_data_loss_warning(self) -> None:
        """1パス処理（アンチパターン）では、aliasが欠損してWARNINGが出ることを確認."""
        from unittest.mock import patch

        csv_df = pl.DataFrame(
            {
                "source_tag": ["witch", "mage"],
                "deprecated_tags": ["sorceress", "wizard"],
            }
        )

        existing_tags: set[str] = set()

        # === Single-Pass (Anti-pattern): Register only source_tag ===
        source_tag_df = csv_df.select("source_tag")
        merged_df = merge_tags(existing_tags, source_tag_df, next_tag_id=1)
        tags_mapping = {row["tag"]: row["tag_id"] for row in merged_df.to_dicts()}

        # tags_mapping contains only witch and mage (no sorceress or wizard)
        assert len(tags_mapping) == 2
        assert "witch" in tags_mapping
        assert "mage" in tags_mapping
        assert "sorceress" not in tags_mapping
        assert "wizard" not in tags_mapping

        # === Try to create TAG_STATUS (will fail for aliases) ===
        with patch("genai_tag_db_dataset_builder.core.merge.logger.warning") as mock_warning:
            records_1 = process_deprecated_tags(
                canonical_tag="witch",
                deprecated_tags="sorceress",
                format_id=1,
                tags_mapping=tags_mapping,
            )

        # WARNING logged because sorceress is missing
        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "Skipped 1 alias(es) missing from TAGS table" in warning_message
        assert "sorceress" in warning_message
        assert "canonical: witch" in warning_message
        assert "Ensure two-pass processing" in warning_message

        # Only canonical record created (alias skipped)
        assert len(records_1) == 1
        assert records_1[0]["alias"] == 0
