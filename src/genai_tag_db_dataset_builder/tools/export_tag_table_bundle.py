"""Export prepared legacy tag-table review files as a portable enrichment bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypedDict

from genai_tag_db_dataset_builder.core.normalize import normalize_tag


class SourceReference(TypedDict):
    source_path: str
    source_row: int


class BundleTag(TypedDict):
    tag: str
    source_tag: str
    translations: list[str]
    source_references: list[SourceReference]


class BundleCategory(TypedDict):
    category_id: str
    namespace: str
    parent_id: str | None
    label: str
    depth: int
    path: list[str]


class BundleAssignment(TypedDict):
    tag: str
    category_id: str
    source_references: list[SourceReference]


class TagTableBundle(TypedDict):
    schema_version: int
    bundle_id: str
    tags: list[BundleTag]
    categories: list[BundleCategory]
    assignments: list[BundleAssignment]


_PROMPT_SYNTAX = re.compile(r"[,\[\]{}<>]|:\s*[-+]?(?:\d|\.\d)|\(\(|\)\)")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _is_simple_tag(tag: str) -> bool:
    """Reject prompt syntax before canonicalization can hide its original form."""
    unescaped = tag.replace(r"\(", "(").replace(r"\)", ")")
    return (
        bool(tag.strip())
        and len(tag) <= 90
        and tag.isascii()
        and not any(ord(character) < 32 or ord(character) == 127 for character in tag)
        and not tag.lstrip().startswith("#")
        and _PROMPT_SYNTAX.search(unescaped) is None
    )


def _reference(value: Any) -> tuple[str, int]:
    item = _object(value, "source reference")
    raw_path = _text(item.get("source_path"), "source_path")
    source_path = raw_path.replace("\\", "/")
    path = PurePosixPath(source_path)
    if (
        path.is_absolute()
        or PureWindowsPath(raw_path).drive
        or ".." in path.parts
        or ":" in source_path
        or any(ord(character) < 32 for character in source_path)
    ):
        raise ValueError("source_path must be a portable relative path")
    row = item.get("source_row")
    if not isinstance(row, int) or isinstance(row, bool) or row < 1:
        raise ValueError("source_row must be a positive integer")
    return path.as_posix(), row


def _references(values: set[tuple[str, int]]) -> list[SourceReference]:
    return [{"source_path": path, "source_row": row} for path, row in sorted(values)]


def _read_tags(review_dir: Path) -> list[BundleTag]:
    candidates = _list(
        json.loads((review_dir / "primary_candidates.json").read_text(encoding="utf-8")),
        "primary candidates",
    )
    translations: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    references: dict[str, set[tuple[str, int]]] = {}
    for value in candidates:
        candidate = _object(value, "tag candidate")
        raw_tag = _text(candidate.get("tag"), "candidate tag")
        if _list(candidate.get("flags"), "candidate flags") or not _is_simple_tag(raw_tag):
            continue
        tag = normalize_tag(raw_tag)
        translations.setdefault(tag, set())
        sources.setdefault(tag, set())
        references.setdefault(tag, set())
        for translation in _list(candidate.get("translations"), "translations"):
            translations[tag].add(_text(translation, "translation").strip())
        provenance = _list(candidate.get("provenance"), "provenance")
        if not provenance:
            raise ValueError(f"Tag {tag!r} has no source references")
        for value in provenance:
            source = _object(value, "provenance record")
            references[tag].add(_reference(source))
            source_tag = _text(source.get("source_tag", raw_tag), "source_tag")
            if not _is_simple_tag(source_tag):
                raise ValueError(f"Tag {tag!r} has a source_tag containing prompt syntax")
            sources[tag].add(source_tag)
    return [
        {
            "tag": tag,
            "source_tag": min(sources[tag]),
            "translations": sorted(translations[tag]),
            "source_references": _references(references[tag]),
        }
        for tag in sorted(translations)
    ]


def _read_categories(review_dir: Path) -> list[BundleCategory]:
    categories: dict[str, BundleCategory] = {}
    with (review_dir / "categories.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            category_id = _text(row.get("category_id"), "category_id")
            path = [
                _text(part, "category path component")
                for part in _list(json.loads(row["path"]), "category path")
            ]
            category: BundleCategory = {
                "category_id": category_id,
                "namespace": _text(row.get("namespace"), "category namespace"),
                "parent_id": row["parent_id"] or None,
                "label": _text(row.get("label"), "category label"),
                "depth": int(row["depth"]),
                "path": path,
            }
            if not path or category["depth"] != len(path) or category["label"] != path[-1]:
                raise ValueError(f"Category {category_id!r} has inconsistent depth, path, or label")
            if category_id in categories and categories[category_id] != category:
                raise ValueError(f"Conflicting category {category_id!r}")
            categories[category_id] = category
    category_paths: set[tuple[str, tuple[str, ...]]] = set()
    for category in categories.values():
        path_key = (category["namespace"], tuple(category["path"]))
        if path_key in category_paths:
            raise ValueError("Duplicate category path")
        category_paths.add(path_key)
        parent_id = category["parent_id"]
        if parent_id is None:
            if category["depth"] != 1:
                raise ValueError(f"Category {category['category_id']!r} is missing its parent")
            continue
        parent = categories.get(parent_id)
        if (
            parent is None
            or parent["namespace"] != category["namespace"]
            or parent["path"] != category["path"][:-1]
        ):
            raise ValueError(f"Category {category['category_id']!r} has an invalid parent")
    return sorted(
        categories.values(), key=lambda item: (item["namespace"], item["path"], item["category_id"])
    )


def _read_assignments(review_dir: Path, tags: set[str], category_ids: set[str]) -> list[BundleAssignment]:
    references: dict[tuple[str, str], set[tuple[str, int]]] = {}
    with (review_dir / "tag_categories.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw_tag = _text(row.get("tag"), "assignment tag")
            if not _is_simple_tag(raw_tag):
                continue
            tag = normalize_tag(raw_tag)
            if tag not in tags:
                continue
            category_id = _text(row.get("category_id"), "assignment category_id")
            if category_id not in category_ids:
                raise ValueError(f"Assignment references unknown category {category_id!r}")
            assignment = references.setdefault((tag, category_id), set())
            values = _list(json.loads(row["source_references"]), "assignment source references")
            assignment.update(_reference(value) for value in values)
    return [
        {"tag": tag, "category_id": category_id, "source_references": _references(values)}
        for (tag, category_id), values in sorted(references.items())
    ]


def build_bundle(review_dir: Path) -> TagTableBundle:
    """Build a deterministic, database-independent bundle from prepared review files."""
    tags = _read_tags(review_dir)
    categories = _read_categories(review_dir)
    assignments = _read_assignments(
        review_dir, {item["tag"] for item in tags}, {item["category_id"] for item in categories}
    )
    return {
        "schema_version": 1,
        "bundle_id": "legacy-tag-table",
        "tags": tags,
        "categories": categories,
        "assignments": assignments,
    }


def export_bundle(review_dir: Path, output: Path) -> dict[str, int | str]:
    """Write UTF-8 JSON with stable ordering and return counts and its SHA-256."""
    bundle = build_bundle(review_dir)
    payload = (json.dumps(bundle, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "tags": len(bundle["tags"]),
        "categories": len(bundle["categories"]),
        "assignments": len(bundle["assignments"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir", type=Path, required=True, help="Prepared tag_table_review directory"
    )
    parser.add_argument("--output", type=Path, required=True, help="Portable JSON bundle to write")
    args = parser.parse_args(argv)
    try:
        result = export_bundle(args.review_dir, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
