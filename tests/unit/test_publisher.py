from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import builder_ci.publisher as publisher


class _FakeApi:
    def __init__(self, token: str | None = None) -> None:
        self.folder_calls: list[dict[str, Any]] = []
        self.file_calls: list[dict[str, Any]] = []

    def upload_folder(self, **kwargs: Any) -> None:
        self.folder_calls.append(kwargs)

    def upload_file(self, **kwargs: Any) -> None:
        self.file_calls.append(kwargs)


def test_publish_deletes_stale_files_in_uploaded_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _FakeApi()
    monkeypatch.setattr(publisher, "HfApi", lambda token=None: api)
    monkeypatch.setattr(publisher, "create_repo", lambda **_k: None)

    out = tmp_path / "out"
    (out / "parquet_danbooru").mkdir(parents=True)
    (out / "report").mkdir()
    db = out / "x.sqlite"
    db.write_bytes(b"x")
    (out / "README.md").write_text("r", encoding="utf-8")
    manifest = out / "build_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    publisher.publish_dataset(
        output_db=db,
        output_dir=out,
        parquet_dir=out / "parquet_danbooru",
        report_dir=out / "report",
        manifest_path=manifest,
        repo_id="owner/repo",
        token="t",
    )

    assert {c["path_in_repo"] for c in api.folder_calls} == {"parquet_danbooru", "report"}
    # フォルダ内の古いファイル (前回ビルドの余分な Parquet シャード等) をリポジトリから消す
    assert all(c["delete_patterns"] == "*" for c in api.folder_calls)
