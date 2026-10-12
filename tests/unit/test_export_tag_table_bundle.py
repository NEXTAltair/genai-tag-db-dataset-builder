"""Portable enrichment export preserves data while removing review-only metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from genai_tag_db_dataset_builder.tools.export_tag_table_bundle import (
    build_bundle,
    export_bundle,
    main,
)


def _candidate(tag: str, row: int = 2, **extra: Any) -> dict[str, Any]:
    return {
        "tag": tag,
        "flags": [],
        "translations": ["長い髪"],
        "provenance": [{"source_tag": tag, "source_path": "table/elements.tsv", "source_row": row}],
        **extra,
    }


def _write_review(
    review_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    categories: list[dict[str, Any]] | None = None,
    assignments: list[dict[str, Any]] | None = None,
) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "primary_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False), encoding="utf-8"
    )
    if categories is None:
        categories = [
            {
                "category_id": "root",
                "namespace": "generator",
                "parent_id": "",
                "label": "身体",
                "depth": 1,
                "path": json.dumps(["身体"], ensure_ascii=False),
            },
            {
                "category_id": "hair",
                "namespace": "generator",
                "parent_id": "root",
                "label": "髪",
                "depth": 2,
                "path": json.dumps(["身体", "髪"], ensure_ascii=False),
            },
        ]
    if assignments is None:
        assignments = [
            {
                "tag": candidate["tag"],
                "category_id": "hair",
                "source_references": json.dumps(candidate["provenance"], ensure_ascii=False),
            }
            for candidate in candidates
        ]
    for name, columns, rows in [
        ("categories.tsv", ["category_id", "namespace", "parent_id", "label", "depth", "path"], categories),
        ("tag_categories.tsv", ["tag", "category_id", "source_references"], assignments),
    ]:
        with (review_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    return review_dir


def test_bundle_contract_preserves_translations_and_hierarchy(tmp_path: Path) -> None:
    review = _write_review(tmp_path, [_candidate("Long_Hair")])

    bundle = build_bundle(review)

    assert set(bundle) == {"schema_version", "bundle_id", "tags", "categories", "assignments"}
    assert bundle["schema_version"] == 1
    assert bundle["bundle_id"] == "legacy-tag-table"
    assert bundle["tags"] == [
        {
            "tag": "long hair",
            "source_tag": "Long_Hair",
            "translations": ["長い髪"],
            "source_references": [{"source_path": "table/elements.tsv", "source_row": 2}],
        }
    ]
    assert bundle["categories"][1] == {
        "category_id": "hair",
        "namespace": "generator",
        "parent_id": "root",
        "label": "髪",
        "depth": 2,
        "path": ["身体", "髪"],
    }
    assert bundle["assignments"][0]["tag"] == "long hair"


def test_canonical_duplicates_merge_tags_assignments_and_source_references(tmp_path: Path) -> None:
    first = _candidate("Long_Hair", translations=["長い髪", "ロングヘア", "長い髪"])
    first["provenance"].append(first["provenance"][0].copy())
    second = _candidate("long hair", row=3, translations=["ロングヘア"])
    review = _write_review(tmp_path, [first, second, first])

    bundle = build_bundle(review)

    assert len(bundle["tags"]) == 1
    assert len(bundle["assignments"]) == 1
    assert bundle["tags"][0]["translations"] == sorted(["長い髪", "ロングヘア"])
    expected = [
        {"source_path": "table/elements.tsv", "source_row": 2},
        {"source_path": "table/elements.tsv", "source_row": 3},
    ]
    assert bundle["tags"][0]["source_references"] == expected
    assert bundle["assignments"][0]["source_references"] == expected


@pytest.mark.parametrize(
    "tag",
    [
        "long hair, red hair",
        "(long hair:1.2)",
        "((long hair))",
        "[long hair]",
        "{long hair}",
        "<lora:model:1>",
        "long hair\nred hair",
        "長い髪",
        "#preset",
        "x" * 91,
    ],
)
def test_prompt_syntax_is_excluded_before_normalization(tmp_path: Path, tag: str) -> None:
    review = _write_review(tmp_path, [_candidate(tag), _candidate("long hair", row=3)])

    bundle = build_bundle(review)

    assert [item["tag"] for item in bundle["tags"]] == ["long hair"]
    assert len(bundle["assignments"]) == 1


def test_flagged_candidates_are_excluded_with_their_assignments(tmp_path: Path) -> None:
    review = _write_review(tmp_path, [_candidate("long hair", flags=["review"]), _candidate("red hair")])

    bundle = build_bundle(review)

    assert [item["tag"] for item in bundle["tags"]] == ["red hair"]
    assert [item["tag"] for item in bundle["assignments"]] == ["red hair"]


def test_parenthesis_tags_and_emoticons_use_existing_normalizer(tmp_path: Path) -> None:
    review = _write_review(tmp_path, [_candidate("bow_(weapon)"), _candidate(":D")])

    bundle = build_bundle(review)

    assert {item["tag"] for item in bundle["tags"]} == {r"bow \(weapon\)", ":D"}


def test_review_metadata_and_private_db_paths_are_not_exported(tmp_path: Path) -> None:
    candidate = _candidate(
        "long hair",
        db_hits=[r"C:\Users\private\cache\database.sqlite"],
        db_translations=["DO NOT EXPORT"],
        preferred_tags=["wrong alias"],
        status="new_tag_candidate",
    )
    candidate["provenance"][0]["duplicate_paths"] = [r"C:\Users\private\other.tsv"]
    candidate["provenance"][0]["raw_categories"] = ["temporary"]
    review = _write_review(tmp_path, [candidate])

    payload = json.dumps(build_bundle(review), ensure_ascii=False)

    for excluded in [
        "private",
        "database.sqlite",
        "DO NOT EXPORT",
        "wrong alias",
        "new_tag_candidate",
        "duplicate_paths",
        "raw_categories",
        "temporary",
    ]:
        assert excluded not in payload


@pytest.mark.parametrize(
    "source_path",
    [
        "/private/table.tsv",
        r"C:\private\table.tsv",
        "../table.tsv",
        "table/../../private.tsv",
        "https://private/table.tsv",
    ],
)
def test_absolute_and_traversal_source_paths_are_rejected(tmp_path: Path, source_path: str) -> None:
    candidate = _candidate("long hair")
    candidate["provenance"][0]["source_path"] = source_path
    review = _write_review(tmp_path, [candidate])

    with pytest.raises(ValueError, match="portable relative path"):
        build_bundle(review)


def test_relative_windows_source_path_is_portable(tmp_path: Path) -> None:
    candidate = _candidate("long hair")
    candidate["provenance"][0]["source_path"] = r"table\elements.tsv"
    review = _write_review(tmp_path, [candidate])

    assert build_bundle(review)["tags"][0]["source_references"][0]["source_path"] == "table/elements.tsv"


def test_unknown_category_assignment_is_rejected(tmp_path: Path) -> None:
    review = _write_review(
        tmp_path,
        [_candidate("long hair")],
        assignments=[{"tag": "long hair", "category_id": "missing", "source_references": "[]"}],
    )

    with pytest.raises(ValueError, match="unknown category"):
        build_bundle(review)


def test_broken_category_parent_is_rejected(tmp_path: Path) -> None:
    review = _write_review(
        tmp_path,
        [_candidate("long hair")],
        categories=[
            {
                "category_id": "hair",
                "namespace": "generator",
                "parent_id": "missing",
                "label": "髪",
                "depth": 2,
                "path": json.dumps(["身体", "髪"], ensure_ascii=False),
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid parent"):
        build_bundle(review)


def test_export_bytes_are_stable_after_input_order_changes(tmp_path: Path) -> None:
    first = _candidate("Long_Hair", row=3, translations=["長い髪", "ロングヘア"])
    second = _candidate("long hair", row=2)
    review = _write_review(tmp_path / "review", [first, second])
    first_path = tmp_path / "first.json"
    first_result = export_bundle(review, first_path)
    first["translations"].reverse()
    _write_review(review, [second, first])
    second_path = tmp_path / "nested" / "second.json"

    second_result = export_bundle(review, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_result == second_result
    assert first_result["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert "長い髪".encode() in first_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")


def test_cli_exports_bundle_and_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    review = _write_review(tmp_path / "review", [_candidate("long hair")])
    output = tmp_path / "bundle.json"

    assert main(["--review-dir", str(review), "--output", str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["tags"] == 1
    assert summary["categories"] == 2
    assert summary["assignments"] == 1
    assert output.is_file()
