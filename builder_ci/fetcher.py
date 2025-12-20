"""外部ソースの取得とリビジョン管理."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from loguru import logger


def load_sources_config(sources_yml: Path) -> list[dict]:
    """sources.ymlを読み込んで有効なソースのリストを返す.

    Args:
        sources_yml: sources.ymlファイルのパス

    Returns:
        有効なソースの辞書のリスト
    """
    with open(sources_yml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sources = config.get("sources", [])

    # enabled=true のもののみを返す
    enabled_sources = [s for s in sources if s.get("enabled", True)]

    logger.info(f"Loaded {len(enabled_sources)} enabled sources from {sources_yml}")
    return enabled_sources


def fetch_github_repo(source: dict, dest: Path, force: bool = False) -> dict:
    """GitHubリポジトリをcloneしてcommit hashを記録.

    Args:
        source: ソース定義辞書
        dest: 保存先ディレクトリ
        force: 既存ディレクトリを削除して再取得するか

    Returns:
        取得結果のメタデータ辞書
    """
    url = source["url"]
    source_id = source["id"]

    logger.info(f"Fetching GitHub repo: {source_id} from {url}")

    # 既存ディレクトリの処理
    if dest.exists():
        if force:
            logger.warning(f"Removing existing directory: {dest}")
            shutil.rmtree(dest)
        else:
            if not (dest / ".git").exists():
                logger.warning(f"Existing path is not a git repo, recreating: {dest}")
                shutil.rmtree(dest)
            else:
                logger.info(f"Updating existing repo: {dest}")
                subprocess.run(
                    ["git", "fetch", "--prune", "--tags", "origin"],
                    cwd=dest,
                    check=True,
                )
                head_ref = subprocess.run(
                    ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                    cwd=dest,
                    capture_output=True,
                    text=True,
                )
                target_ref = (
                    head_ref.stdout.strip()
                    if head_ref.returncode == 0 and head_ref.stdout.strip()
                    else "refs/remotes/origin/main"
                )
                subprocess.run(
                    ["git", "reset", "--hard", target_ref],
                    cwd=dest,
                    check=True,
                )
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=dest,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                commit_hash = result.stdout.strip()

                return {
                    "id": source_id,
                    "kind": "github",
                    "url": url,
                    "commit_hash": commit_hash,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "skipped": False,
                }

    # git clone
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", url, str(dest)], check=True)

    # commit hash取得
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dest,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_hash = result.stdout.strip()

    logger.info(f"Cloned {source_id} at commit {commit_hash[:8]}")

    return {
        "id": source_id,
        "kind": "github",
        "url": url,
        "commit_hash": commit_hash,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "skipped": False,
    }


def fetch_hf_dataset(source: dict, dest: Path, force: bool = False) -> dict:
    """HF datasetを取得してrevisionを記録.

    Args:
        source: ソース定義辞書
        dest: 保存先ディレクトリ
        force: 既存ディレクトリを削除して再取得するか

    Returns:
        取得結果のメタデータ辞書
    """
    repo_id = source["repo_id"]
    source_id = source["id"]

    logger.info(f"Fetching HF dataset: {source_id} from {repo_id}")

    hf_api = HfApi()
    revision = None
    try:
        try:
            info = hf_api.dataset_info(repo_id, repo_type="dataset")
        except TypeError:
            info = hf_api.dataset_info(repo_id)
        revision = info.sha
    except Exception as exc:
        logger.warning(f"Failed to resolve HF revision for {repo_id}: {exc}")

    # use_datasets_api=trueの場合はbuilder.pyが直接取得するのでスキップ
    if source.get("hf_config", {}).get("use_datasets_api"):
        logger.info(f"Skipping fetch for {source_id} (use_datasets_api=true)")
        return {
            "id": source_id,
            "kind": "hf_dataset",
            "repo_id": repo_id,
            "revision": revision or "unknown",
            "fetch_method": "datasets_api",
            "note": "Fetched directly by builder.py via datasets.load_dataset()",
            "skipped": True,
        }

    # 既存ディレクトリの処理
    if dest.exists() and not force:
        sha_path = dest / ".hf_sha"
        if revision and sha_path.exists():
            existing_sha = sha_path.read_text(encoding="utf-8").strip()
            if existing_sha == revision:
                logger.info(f"Directory already exists with matching revision, skipping download: {dest}")
                return {
                    "id": source_id,
                    "kind": "hf_dataset",
                    "repo_id": repo_id,
                    "revision": revision,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "skipped": True,
                }
        if revision is None:
            logger.warning(f"Revision unavailable; keeping existing download without update: {dest}")
            return {
                "id": source_id,
                "kind": "hf_dataset",
                "repo_id": repo_id,
                "revision": "unknown",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "skipped": True,
            }
        logger.info(f"Existing download is stale or missing sha, re-downloading: {dest}")
        shutil.rmtree(dest)

    # snapshot_downloadで取得（SQLite等のバイナリファイル用）
    dest.parent.mkdir(parents=True, exist_ok=True)

    allow_patterns = source.get("paths_include", ["*"])
    logger.info(f"Downloading {repo_id} with patterns: {allow_patterns}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest,
            allow_patterns=allow_patterns,
            revision=revision,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            allow_patterns=allow_patterns,
            revision=revision,
        )

    if revision:
        (dest / ".hf_sha").write_text(revision, encoding="utf-8")

    logger.info(f"Downloaded {source_id} at revision {revision[:8] if revision else 'unknown'}")

    return {
        "id": source_id,
        "kind": "hf_dataset",
        "repo_id": repo_id,
        "revision": revision or "unknown",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "skipped": False,
    }


def fetch_all_sources(sources_yml: Path, external_sources_dir: Path, force: bool = False) -> list[dict]:
    """全ソースを取得してメタデータリストを返す.

    Args:
        sources_yml: sources.ymlファイルのパス
        external_sources_dir: 外部ソースを保存するベースディレクトリ
        force: 既存ディレクトリを削除して再取得するか

    Returns:
        取得結果のメタデータ辞書のリスト
    """
    sources = load_sources_config(sources_yml)
    results = []

    for source in sources:
        source_id = source["id"]
        kind = source["kind"]
        dest = external_sources_dir / source_id

        try:
            if kind == "github":
                result = fetch_github_repo(source, dest, force=force)
            elif kind == "hf_dataset":
                result = fetch_hf_dataset(source, dest, force=force)
            else:
                logger.warning(f"Unknown source kind: {kind} for {source_id}")
                continue

            # paths_includeを記録
            result["paths_include"] = source.get("paths_include", [])
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to fetch {source_id}: {e}")
            # エラーでも記録は残す
            results.append(
                {
                    "id": source_id,
                    "kind": kind,
                    "error": str(e),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "skipped": False,
                }
            )

    logger.info(f"Fetched {len(results)} sources")
    return results
