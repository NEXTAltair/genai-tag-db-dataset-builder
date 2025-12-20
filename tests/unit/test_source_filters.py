from __future__ import annotations

from pathlib import Path

from genai_tag_db_dataset_builder import builder


def test_should_include_source_no_filters() -> None:
    assert builder._should_include_source(
        "TagDB_DataSource_CSV/A/foo.csv",
        include_exact=set(),
        include_patterns=[],
        exclude_exact=set(),
        exclude_patterns=[],
    )


def test_should_include_source_include_exact() -> None:
    assert builder._should_include_source(
        "TagDB_DataSource_CSV/A/foo.csv",
        include_exact={"TagDB_DataSource_CSV/A/foo.csv"},
        include_patterns=[],
        exclude_exact=set(),
        exclude_patterns=[],
    )
    assert not builder._should_include_source(
        "TagDB_DataSource_CSV/A/bar.csv",
        include_exact={"TagDB_DataSource_CSV/A/foo.csv"},
        include_patterns=[],
        exclude_exact=set(),
        exclude_patterns=[],
    )


def test_should_include_source_include_pattern_and_exclude_pattern() -> None:
    assert builder._should_include_source(
        "TagDB_DataSource_CSV/A/foo.csv",
        include_exact=set(),
        include_patterns=["TagDB_DataSource_CSV/A/*.csv"],
        exclude_exact=set(),
        exclude_patterns=["*/skip_*.csv"],
    )
    assert not builder._should_include_source(
        "TagDB_DataSource_CSV/A/skip_foo.csv",
        include_exact=set(),
        include_patterns=["TagDB_DataSource_CSV/A/*.csv"],
        exclude_exact=set(),
        exclude_patterns=["*/skip_*.csv"],
    )


def test_infer_source_timestamp_utc_midnight() -> None:
    assert (
        builder._infer_source_timestamp_utc_midnight(Path("danbooru_241016.csv"))
        == "2024-10-16 00:00:00+00:00"
    )
    assert (
        builder._infer_source_timestamp_utc_midnight(Path("danbooru_20241016.csv"))
        == "2024-10-16 00:00:00+00:00"
    )
    assert builder._infer_source_timestamp_utc_midnight(Path("no_date.csv")) is None


def test_is_authoritative_count_source() -> None:
    assert builder._is_authoritative_count_source(Path("danbooru_241016.csv"))
    assert not builder._is_authoritative_count_source(Path("danbooru.csv"))
    assert not builder._is_authoritative_count_source(Path("e621_241016.csv"))
