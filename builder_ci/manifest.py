"""ビルドマニフェスト（build_manifest.json）の生成と管理."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# override ファイルが存在しない場合の安定したハッシュ値。
# （None と "未設定" を区別し、初回 manifest との比較を安定させる）
OVERRIDE_HASH_ABSENT = "absent"


def create_build_manifest(
    version: str,
    target: str,
    base_db_info: dict | None,
    sources_metadata: list[dict],
    statistics: dict | None = None,
    health_checks: dict | None = None,
    builder_version: str = "0.1.0",
    override_hash: str | None = None,
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
        override_hash: overrides/alias_resolution.yml のハッシュ（再ビルド判定に使用）

    Returns:
        マニフェスト辞書
    """
    manifest = {
        "build_info": {
            "version": version,
            "target": target,
            "built_at": datetime.now(UTC).isoformat(),
            "builder_version": builder_version,
            "override_hash": override_hash if override_hash is not None else OVERRIDE_HASH_ABSENT,
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


def compute_override_hash(override_path: Path | None) -> str:
    """override ファイルの内容ハッシュを返す.

    ファイルが無い場合は :data:`OVERRIDE_HASH_ABSENT` を返し、manifest 比較が
    安定するようにする。

    Args:
        override_path: overrides/alias_resolution.yml のパス（None 可）

    Returns:
        ``sha256:<hex>`` 形式のハッシュ、または "absent"
    """
    if override_path is None:
        return OVERRIDE_HASH_ABSENT
    override_path = Path(override_path)
    if not override_path.exists():
        return OVERRIDE_HASH_ABSENT
    digest = hashlib.sha256(override_path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def get_manifest_override_hash(manifest: dict) -> str:
    """manifest に記録された override hash を取得する（無ければ absent 扱い）."""
    value = manifest.get("build_info", {}).get("override_hash")
    return value if isinstance(value, str) else OVERRIDE_HASH_ABSENT


def restore_build_manifest_from_hf(repo_id: str, manifest_path: Path) -> bool:
    """HF dataset 上の build_manifest.json をローカルへ復元する.

    手動・定期どちらの workflow でも、既存 manifest を取り込んでから差分判定する
    ために使う。取得に失敗した場合（repo 未作成 / ファイル無し / ネットワーク等）は
    False を返し、呼び出し側で初回ビルド扱いに委ねる。

    Args:
        repo_id: 対象 HF dataset repo id
        manifest_path: 復元先ローカルパス

    Returns:
        復元に成功したら True
    """
    # huggingface_hub は重いので遅延 import（テスト時の依存も最小化）。
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="build_manifest.json",
            repo_type="dataset",
        )
    except (EntryNotFoundError, RepositoryNotFoundError) as exc:
        logger.info(f"No existing manifest on HF for {repo_id} (treating as first build): {exc}")
        return False
    except Exception as exc:
        logger.warning(f"Failed to restore manifest from HF for {repo_id}: {exc}")
        return False

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, manifest_path)
    logger.info(f"Restored build_manifest.json from HF: {repo_id} -> {manifest_path}")
    return True


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
