"""Unit tests for normalize utilities."""

from genai_tag_db_dataset_builder.core.normalize import (
    canonicalize_source_tag,
    is_kaomoji,
    normalize_tag,
)


class TestNormalizeTag:
    def test_lowercase_conversion(self) -> None:
        assert normalize_tag("Witch") == "witch"
        assert normalize_tag("HATSUNE_MIKU") == "hatsune miku"

    def test_underscore_to_space(self) -> None:
        assert normalize_tag("spiked_collar") == "spiked collar"
        assert normalize_tag("1_girl") == "1 girl"

    def test_strip_whitespace(self) -> None:
        assert normalize_tag("  tag  ") == "tag"
        assert normalize_tag(" multiple words ") == "multiple words"

    def test_preserve_kaomoji(self) -> None:
        assert normalize_tag(":D") == ":D"
        assert normalize_tag("(o_o)") == r"\(o_o\)"


class TestCanonicalizeSourceTag:
    def test_lowercase_only(self) -> None:
        assert canonicalize_source_tag("Witch") == "witch"
        assert canonicalize_source_tag("Spiked_Collar") == "spiked_collar"

    def test_preserve_kaomoji_case(self) -> None:
        assert canonicalize_source_tag(":D") == ":D"
        assert canonicalize_source_tag("(o_o)") == "(o_o)"


class TestIsKaomoji:
    def test_scoring(self) -> None:
        assert is_kaomoji("^_^") is True
        assert is_kaomoji(":D") is True
        assert is_kaomoji("(o_o)") is True
        assert is_kaomoji("spiked_collar") is False
        assert is_kaomoji("long hair") is False
