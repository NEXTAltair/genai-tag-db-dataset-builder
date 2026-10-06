"""CI Hugging Face publisher."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo
from loguru import logger


def _upload_file(
    api: HfApi,
    path: Path,
    repo_id: str,
    commit_message: str | None,
) -> None:
    if not path.exists():
        logger.warning(f"Missing file, skipping upload: {path}")
        return
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def _upload_folder(
    api: HfApi,
    folder_path: Path,
    repo_id: str,
    commit_message: str | None,
) -> None:
    if not folder_path.exists():
        logger.warning(f"Missing folder, skipping upload: {folder_path}")
        return
    # delete_patterns="*": リポジトリ側のこのフォルダにだけ残っている古いファイルを削除する。
    # (これが無いと、シャード数が減ったビルドで過去のビルドの parquet_danbooru/*.parquet が
    #  残り続け、Dataset Viewer や利用者に古い (混入を含む) データが見えてしまう)
    api.upload_folder(
        folder_path=str(folder_path),
        path_in_repo=folder_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        delete_patterns="*",
    )


def publish_dataset(
    output_db: Path,
    output_dir: Path,
    parquet_dir: Path,
    report_dir: Path,
    manifest_path: Path,
    repo_id: str,
    token: str,
    commit_message: str | None = None,
) -> None:
    api = HfApi(token=token)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )

    logger.info(f"Uploading dataset to Hugging Face: {repo_id}")
    _upload_file(api, output_db, repo_id, commit_message)
    _upload_file(api, output_dir / "README.md", repo_id, commit_message)
    _upload_file(api, manifest_path, repo_id, commit_message)
    _upload_folder(api, parquet_dir, repo_id, commit_message)
    _upload_folder(api, report_dir, repo_id, commit_message)

    logger.info(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
