#!/usr/bin/env python3
"""Phase 8.1 基盤構築のローカルテスト."""

from pathlib import Path

from builder_ci import (
    create_build_manifest,
    load_sources_config,
    write_build_manifest,
)

# sources.ymlのパス
sources_yml = Path(__file__).parent / "builder_ci" / "sources.yml"
test_output = Path(__file__).parent / "test_output"
test_output.mkdir(exist_ok=True)

print("=" * 60)
print("Phase 8.1: 基盤構築テスト")
print("=" * 60)

# 1. sources.ymlの読み込みテスト
print("\n[1] sources.yml読み込みテスト")
sources = load_sources_config(sources_yml)
print(f"   有効なソース数: {len(sources)}")
for src in sources:
    print(f"   - {src['id']} ({src['license']}) - applies_to: {src['applies_to']}")

# 2. manifest作成テスト
print("\n[2] build_manifest.json生成テスト")

# ダミーのソースメタデータ
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

# ダミーのベースDB情報
base_db_info = {
    "repo_id": "NEXTAltair/genai-image-tag-db-mit",
    "revision": "test_revision_123",
    "downloaded_at": "2025-12-19T11:30:00+00:00",
}

# マニフェスト作成
manifest = create_build_manifest(
    version="v4.2.0-test",
    target="mit",
    base_db_info=base_db_info,
    sources_metadata=sources_metadata,
    statistics={"total_sources": 2, "total_tags": 1000000},
    health_checks={"foreign_key_violations": 0, "status": "PASSED"},
)

# マニフェスト保存
manifest_path = test_output / "build_manifest_test.json"
write_build_manifest(manifest, manifest_path)
print(f"   マニフェスト保存: {manifest_path}")
print(f"   - バージョン: {manifest['build_info']['version']}")
print(f"   - ターゲット: {manifest['build_info']['target']}")
print(f"   - ソース数: {len(manifest['sources'])}")

# 3. モジュールインポート確認
print("\n[3] モジュールインポート確認")
try:
    from builder_ci import __version__

    print(f"   builder_ciバージョン: {__version__}")
    print("   ✓ 全てのモジュールが正常にインポートされました")
except ImportError as e:
    print(f"   ✗ インポートエラー: {e}")

print("\n" + "=" * 60)
print("Phase 8.1 基盤構築テスト完了")
print("=" * 60)
print(f"\nテスト出力ディレクトリ: {test_output}")
