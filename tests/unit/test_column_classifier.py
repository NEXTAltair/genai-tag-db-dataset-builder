"""Unit tests for tag column classifier."""

import polars as pl
import pytest

from genai_tag_db_dataset_builder.core.column_classifier import (
    TagColumnType,
    calculate_normalize_change_ratio,
    calculate_underscore_ratio,
    classify_tag_column,
    detect_escaped_parentheses,
)


class TestCalculateUnderscoreRatio:
    def test_all_with_underscores(self) -> None:
        tags = ["witch_hat", "long_hair", "red_eyes"]
        assert calculate_underscore_ratio(tags) == 1.0

    def test_none_with_underscores(self) -> None:
        tags = ["witch hat", "long hair", "red eyes"]
        assert calculate_underscore_ratio(tags) == 0.0

    def test_partial_underscores(self) -> None:
        tags = ["witch_hat", "long hair", "red_eyes", "blue dress"]
        assert calculate_underscore_ratio(tags) == 0.5

    def test_empty_list(self) -> None:
        assert calculate_underscore_ratio([]) == 0.0


class TestDetectEscapedParentheses:
    def test_with_escaped_parentheses(self) -> None:
        tags = [r"witch \(hat\)", r"long \(hair\)", "red_eyes"]
        assert detect_escaped_parentheses(tags) == pytest.approx(2 / 3)

    def test_without_escaped_parentheses(self) -> None:
        tags = ["witch hat", "long hair", "red eyes"]
        assert detect_escaped_parentheses(tags) == 0.0

    def test_partial_escaped_parentheses(self) -> None:
        tags = [r"witch \(hat\)", "long hair", "red eyes", "blue dress"]
        assert detect_escaped_parentheses(tags) == 0.25

    def test_empty_list(self) -> None:
        assert detect_escaped_parentheses([]) == 0.0


class TestCalculateNormalizeChangeRatio:
    def test_all_change_on_normalization(self) -> None:
        tags = ["witch_hat", "long_hair", "red_eyes"]
        assert calculate_normalize_change_ratio(tags) == 1.0

    def test_none_change_on_normalization(self) -> None:
        tags = ["witch hat", "long hair", "red eyes"]
        assert calculate_normalize_change_ratio(tags) == 0.0

    def test_partial_change_on_normalization(self) -> None:
        tags = ["witch_hat", "long hair", "red_eyes", "blue dress"]
        assert calculate_normalize_change_ratio(tags) == 0.5

    def test_empty_list(self) -> None:
        assert calculate_normalize_change_ratio([]) == 0.0


class TestClassifyTagColumn:
    def test_classify_source_tags(self) -> None:
        df = pl.DataFrame(
            {
                "tag": [
                    r"witch_\(hat\)",
                    "long_hair",
                    r"red_\(eyes\)",
                    "blue_dress",
                    "white_background",
                ]
            }
        )
        tag_type, signals = classify_tag_column(df, "tag")
        assert tag_type == TagColumnType.SOURCE
        assert signals["source_signals"] >= 2
        assert signals["underscore_ratio"] == 1.0
        assert signals["normalize_change_ratio"] == 1.0

    def test_classify_normalized_tags(self) -> None:
        df = pl.DataFrame(
            {
                "tag": [
                    "witch hat",
                    "long hair",
                    "red eyes",
                    "blue dress",
                    "white background",
                ]
            }
        )
        tag_type, signals = classify_tag_column(df, "tag")
        assert tag_type == TagColumnType.NORMALIZED
        assert signals["normalized_signals"] >= 2
        assert signals["underscore_ratio"] == 0.0
        assert signals["normalize_change_ratio"] == 0.0

    def test_classify_unknown_tags_ambiguous_case(self) -> None:
        df = pl.DataFrame({"tag": ["witch_hat", "long hair", "red_eyes", "blue dress"]})
        tag_type, signals = classify_tag_column(df, "tag")
        assert tag_type == TagColumnType.UNKNOWN
        assert signals["confidence"] == "low"
        assert signals["underscore_ratio"] == 0.5
        assert signals["normalize_change_ratio"] == 0.5

    def test_classify_with_custom_thresholds(self) -> None:
        df = pl.DataFrame({"tag": ["witch_hat", "long hair"]})
        tag_type, signals = classify_tag_column(df, "tag", thresholds={"underscore_threshold": 0.4})
        assert tag_type == TagColumnType.SOURCE
        assert signals["underscore_ratio"] == 0.5

    def test_classify_nonexistent_column_raises_error(self) -> None:
        df = pl.DataFrame({"source_tag": ["witch_hat"]})
        with pytest.raises(ValueError, match="Column 'tag' not found"):
            classify_tag_column(df, "tag")

    def test_classify_empty_column_raises_error(self) -> None:
        df = pl.DataFrame({"tag": []}, schema={"tag": pl.Utf8})
        with pytest.raises(ValueError, match="Column 'tag' is empty"):
            classify_tag_column(df, "tag")

