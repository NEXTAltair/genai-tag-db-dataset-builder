"""ビルドマニフェスト（build_manifest.json）の生成と管理."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def create_build_manifest(
    version: str,
    target: str,
    base_db_info: dict | None,
    sources_metadata: list[dict],
    statistics: dict | None = None,
    health_checks: dict | None = None,
    builder_version: str = "0.1.0",
) -> dict:
    """ビルドマニフェストを作成.

    Args:
        version: データセットバージョン（例: "v4.2.0"）
        target: ビルドターゲット（"cc0", "mit", "cc4", "all"）
        base_db_info: ベースDBの情報（repo_id、revision、downloaded_at）
        sources_metadata: 外部ソースのメタデータリスト（fetch結果）
        statistics: ビルド統計（total_tags、total_translations等）
        health_checks: 健全性チェック結果
        builder_version: builder_ciのバージョン

    Returns:
        マニフェスト辞書
    """
    manifest = {
        "build_info": {
            "version": version,
            "target": target,
            "built_at": datetime.now(UTC).isoformat(),
            "builder_version": builder_version,
        },
        "sources": sources_metadata,
    }

    # base_db情報を追加
    if base_db_info:
        manifest["build_info"]["base_db"] = base_db_info

    # 統計情報を追加
    if statistics:
        manifest["statistics"] = statistics

    # 健全性チェック結果を追加
    if health_checks:
        manifest["health_checks"] = health_checks

    return manifest


def write_build_manifest(manifest: dict, output_path: Path) -> None:
    """マニフェストをJSONファイルとして保存.

    Args:
        manifest: マニフェスト辞書
        output_path: 出力JSONファイルパス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Build manifest written to {output_path}")


def load_build_manifest(manifest_path: Path) -> dict:
    """既存のマニフェストを読み込む.

    Args:
        manifest_path: マニフェストJSONファイルパス

    Returns:
        マニフェスト辞書
    """
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return {}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest from {manifest_path}")
    return manifest


def compare_source_revisions(old_manifest: dict, new_sources_metadata: list[dict]) -> dict:
    """旧マニフェストと新ソースメタデータを比較してリビジョン変更を検出.

    Args:
        old_manifest: 旧マニフェスト辞書
        new_sources_metadata: 新しいソースメタデータリスト

    Returns:
        変更情報の辞書
            {
                "changed": [{"id": ..., "old_revision": ..., "new_revision": ...}, ...],
                "added": [{"id": ...}, ...],
                "removed": [{"id": ...}, ...],
                "unchanged": [{"id": ...}, ...],
            }
    """
    old_sources = {s["id"]: s for s in old_manifest.get("sources", [])}
    new_sources = {s["id"]: s for s in new_sources_metadata}

    changed = []
    added = []
    removed = []
    unchanged = []

    # 新ソースとの比較
    for source_id, new_src in new_sources.items():
        if source_id not in old_sources:
            added.append({"id": source_id})
        else:
            old_src = old_sources[source_id]
            # リビジョン/commit_hashを比較
            old_rev = old_src.get("revision") or old_src.get("commit_hash")
            new_rev = new_src.get("revision") or new_src.get("commit_hash")

            if old_rev != new_rev:
                changed.append(
                    {
                        "id": source_id,
                        "old_revision": old_rev,
                        "new_revision": new_rev,
                    }
                )
            else:
                unchanged.append({"id": source_id})

    # 削除されたソース
    for source_id in old_sources:
        if source_id not in new_sources:
            removed.append({"id": source_id})

    logger.info(
        f"Revision comparison: {len(changed)} changed, {len(added)} added, "
        f"{len(removed)} removed, {len(unchanged)} unchanged"
    )

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "unchanged": unchanged,
    }


def should_rebuild(comparison: dict, force: bool = False) -> bool:
    """リビジョン比較結果からリビルドが必要かを判定.

    Args:
        comparison: compare_source_revisionsの結果
        force: 強制リビルドフラグ

    Returns:
        リビルドが必要ならTrue
    """
    if force:
        logger.info("Force rebuild enabled")
        return True

    has_changes = len(comparison["changed"]) > 0 or len(comparison["added"]) > 0
    if has_changes:
        logger.info("Rebuild required due to source changes")
        return True

    logger.info("No changes detected, rebuild not required")
    return False
