"""Unit tests for build manifest override hash and HF restore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from builder_ci.manifest import (
    OVERRIDE_HASH_ABSENT,
    compute_override_hash,
    create_build_manifest,
    get_manifest_override_hash,
    restore_build_manifest_from_hf,
)


class TestOverrideHash:
    def test_none_path_is_absent(self) -> None:
        assert compute_override_hash(None) == OVERRIDE_HASH_ABSENT

    def test_missing_file_is_absent(self, tmp_path: Path) -> None:
        assert compute_override_hash(tmp_path / "nope.yml") == OVERRIDE_HASH_ABSENT

    def test_present_file_is_stable_and_content_sensitive(self, tmp_path: Path) -> None:
        p = tmp_path / "alias_resolution.yml"
        p.write_text("alias_resolution: {}\n", encoding="utf-8")
        h1 = compute_override_hash(p)
        h2 = compute_override_hash(p)
        assert h1 == h2
        assert h1.startswith("sha256:")

        p.write_text("alias_resolution:\n  zerochan.net:\n    a: B\n", encoding="utf-8")
        assert compute_override_hash(p) != h1

    def test_manifest_records_override_hash(self) -> None:
        manifest = create_build_manifest(
            version="v1",
            target="cc0",
            base_db_info=None,
            sources_metadata=[],
            override_hash="sha256:abc",
        )
        assert manifest["build_info"]["override_hash"] == "sha256:abc"
        assert get_manifest_override_hash(manifest) == "sha256:abc"

    def test_manifest_default_override_hash_is_absent(self) -> None:
        manifest = create_build_manifest(
            version="v1",
            target="cc0",
            base_db_info=None,
            sources_metadata=[],
        )
        assert get_manifest_override_hash(manifest) == OVERRIDE_HASH_ABSENT

    def test_get_override_hash_legacy_manifest(self) -> None:
        # 旧 manifest（override_hash 列が無い）でも absent 扱いになる
        assert get_manifest_override_hash({"build_info": {}}) == OVERRIDE_HASH_ABSENT
        assert get_manifest_override_hash({}) == OVERRIDE_HASH_ABSENT


class TestRestoreManifestFromHf:
    def test_restore_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # HF からダウンロードした体の manifest ファイルを用意
        src = tmp_path / "downloaded_manifest.json"
        src.write_text(json.dumps({"build_info": {"version": "v9"}}), encoding="utf-8")

        def fake_download(repo_id: str, filename: str, repo_type: str) -> str:
            assert filename == "build_manifest.json"
            return str(src)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        dest = tmp_path / "out" / "build_manifest.json"
        ok = restore_build_manifest_from_hf("NEXTAltair/genai-image-tag-db", dest)
        assert ok is True
        assert dest.exists()
        assert json.loads(dest.read_text(encoding="utf-8"))["build_info"]["version"] == "v9"

    def test_restore_missing_entry_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from huggingface_hub.utils import EntryNotFoundError

        def fake_download(repo_id: str, filename: str, repo_type: str) -> str:
            raise EntryNotFoundError("missing")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        dest = tmp_path / "out" / "build_manifest.json"
        ok = restore_build_manifest_from_hf("NEXTAltair/genai-image-tag-db", dest)
        assert ok is False
        assert not dest.exists()

    def test_restore_unexpected_error_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(repo_id: str, filename: str, repo_type: str) -> str:
            raise RuntimeError("network down")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        dest = tmp_path / "out" / "build_manifest.json"
        ok = restore_build_manifest_from_hf("NEXTAltair/genai-image-tag-db", dest)
        assert ok is False
