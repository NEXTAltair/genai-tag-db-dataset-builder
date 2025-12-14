"""Unit tests for tag normalization."""

import pytest
from genai_tag_db_dataset_builder.core.normalize import is_kaomoji, normalize_tag


class TestNormalizeTag:
    """normalize_tag関数のテスト."""

    def test_lowercase_conversion(self) -> None:
        """大文字→小文字変換のテスト."""
        assert normalize_tag("Witch") == "witch"
        assert normalize_tag("HATSUNE_MIKU") == "hatsune miku"

    def test_underscore_to_space(self) -> None:
        """アンダースコア→スペース変換のテスト."""
        assert normalize_tag("spiked_collar") == "spiked collar"
        assert normalize_tag("1_girl") == "1 girl"

    def test_strip_whitespace(self) -> None:
        """前後の空白削除のテスト."""
        assert normalize_tag("  tag  ") == "tag"
        assert normalize_tag(" multiple words ") == "multiple words"

    def test_combined_transformations(self) -> None:
        """複合的な変換のテスト."""
        assert normalize_tag("Spiked_Collar") == "spiked collar"
        assert normalize_tag("  Long_Tag_Name  ") == "long tag name"

    def test_already_normalized(self) -> None:
        """既に正規化済みのタグのテスト."""
        assert normalize_tag("witch") == "witch"
        assert normalize_tag("spiked collar") == "spiked collar"

    def test_numeric_tags(self) -> None:
        """数値を含むタグのテスト."""
        assert normalize_tag("1girl") == "1girl"
        assert normalize_tag("2_girls") == "2 girls"

    def test_special_characters(self) -> None:
        """特殊文字を含むタグのテスト. 顔文字は正則化から除外. ()はエスケープする"""
        assert normalize_tag(":D") == ":D"  # 顔文字は正規化から除外
        assert normalize_tag("(o_o)") == r"\(o_o\)"  # 顔文字は正規化から除外、括弧エスケープのみ

    def test_is_kaomoji_scoring(self) -> None:
        assert is_kaomoji("^_^") is True
        assert is_kaomoji("T_T") is False  # _FACE_UNDERSCORE は小文字を要求するため False
        assert is_kaomoji(":D") is True
        assert is_kaomoji(":d") is True  # IGNORECASE 対応
        assert is_kaomoji("(o_o)") is True
        assert is_kaomoji("spiked_collar") is False
        assert is_kaomoji("long hair") is False
