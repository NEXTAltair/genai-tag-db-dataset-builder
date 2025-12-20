#!/usr/bin/env python3
"""Phase 8.1 scaffold test."""

from __future__ import annotations

from pathlib import Path

from builder_ci import (
    __version__,
    create_build_manifest,
    load_sources_config,
    write_build_manifest,
)


def main() -> None:
    repo_root = Path(__file__).parent
    sources_yml = repo_root / "builder_ci" / "sources.yml"
    test_output = repo_root / "test_output"
    test_output.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 8.1 scaffold test")
    print("=" * 60)

    print("\n[1] sources.yml load")
    sources = load_sources_config(sources_yml)
    print(f"  enabled sources: {len(sources)}")
    for src in sources:
        print(
            f"  - {src['id']} ({src.get('license', 'unknown')}) applies_to={src.get('applies_to', [])}"
        )

    print("\n[2] build_manifest.json write")
    sources_metadata = [
        {
            "id": "p1atdev-danbooru-ja-tag-pair",
            "kind": "hf_dataset",
            "repo_id": "p1atdev/danbooru-ja-tag-pair-20241015",
            "fetch_method": "datasets_api",
            "note": "Fetched directly by builder.py",
            "skipped": True,
            "paths_include": ["data/train-*.parquet"],
        },
        {
            "id": "booru-japanese-tag",
            "kind": "github",
            "url": "https://github.com/boorutan/booru-japanese-tag",
            "commit_hash": "035c0d63cbf70f6a3d8da4fbef31a122b48a9814",
            "fetched_at": "2025-12-19T12:00:00+00:00",
            "skipped": False,
            "paths_include": ["danbooru-machine-jp.csv"],
        },
    ]

    base_db_info = {
        "repo_id": "NEXTAltair/genai-image-tag-db-mit",
        "revision": "test_revision_123",
        "downloaded_at": "2025-12-19T11:30:00+00:00",
    }

    manifest = create_build_manifest(
        version="v4.2.0-test",
        target="mit",
        base_db_info=base_db_info,
        sources_metadata=sources_metadata,
        statistics={"total_sources": 2, "total_tags": 1000000},
        health_checks={"foreign_key_violations": 0, "status": "PASSED"},
    )

    manifest_path = test_output / "build_manifest_test.json"
    write_build_manifest(manifest, manifest_path)
    print(f"  manifest written: {manifest_path}")
    print(f"  - version: {manifest['build_info']['version']}")
    print(f"  - target: {manifest['build_info']['target']}")
    print(f"  - sources: {len(manifest['sources'])}")

    print("\n[3] module import")
    print(f"  builder_ci version: {__version__}")

    print("\n" + "=" * 60)
    print("Phase 8.1 scaffold test done")
    print("=" * 60)
    print(f"\noutput dir: {test_output}")


if __name__ == "__main__":
    main()
