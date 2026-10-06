from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import builder_ci.main as ci_main
from builder_ci.manifest import create_build_manifest, write_build_manifest


class _StopBuildError(Exception):
    """build_dataset に到達したことを示すための番兵."""


def _run_build_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    cc0_revision: str,
) -> bool:
    """既存 manifest (cc0 ソース rev=old) がある状態で派生ビルドを走らせ、再ビルドされたかを返す."""
    out_dir = tmp_path / "out_db_mit"
    out_dir.mkdir()
    target = ci_main.TargetConfig(
        name="mit",
        repo_id="repo",
        output_dir=out_dir,
        output_db=out_dir / "mit.sqlite",
        parquet_dir=out_dir / "parquet",
        report_dir=out_dir / "report",
        manifest_path=out_dir / "build_manifest.json",
    )
    mit_src = {"id": "mit-src", "kind": "hf_dataset"}
    cc0_src = {"id": "cc0-src", "kind": "hf_dataset"}
    revisions = {"mit-src": "m1", "cc0-src": cc0_revision}

    monkeypatch.setattr(ci_main, "_current_builder_version", lambda _root: "bv")
    monkeypatch.setattr(
        ci_main,
        "_fetch_sources",
        lambda srcs, _dir, force=False: [{"id": s["id"], "revision": revisions[s["id"]]} for s in srcs],
    )
    monkeypatch.setattr(ci_main, "_stage_translation_csvs", lambda *_a, **_k: [])
    monkeypatch.setattr(ci_main, "_generate_include_filter", lambda *_a, **_k: tmp_path / "inc.txt")
    monkeypatch.setattr(ci_main, "_hf_translation_datasets", lambda _s: [])
    monkeypatch.setattr(ci_main, "_hf_zh_translation_datasets", lambda _s: [])
    monkeypatch.setattr(ci_main, "_hf_wiki_multilang_datasets", lambda _s: [])

    def _build_dataset(**_kwargs: Any) -> None:
        raise _StopBuildError

    monkeypatch.setattr(ci_main, "build_dataset", _build_dataset)

    base_info = {"repo_id": "cc0_local", "path": "x"}
    manifest = create_build_manifest(
        version="v",
        target="mit",
        base_db_info=base_info,
        sources_metadata=[
            {"id": "mit-src", "revision": "m1"},
            {"id": "inherited:cc0-src", "revision": "old"},
        ],
        statistics={},
        health_checks={},
        builder_version="bv",
        override_hash=ci_main.compute_override_hash(None),
    )
    write_build_manifest(manifest, target.manifest_path)

    try:
        ci_main._build_target(
            target=target,
            sources=[mit_src],
            sources_dir=tmp_path,
            external_sources_dir=tmp_path / "ext",
            base_db_path=tmp_path / "base.sqlite",
            base_db_info=base_info,
            sources_yml=tmp_path / "sources.yml",
            version="v",
            force=False,
            publish=False,
            publish_repo_id=None,
            inherited_sources=[cc0_src],
        )
    except _StopBuildError:
        return True
    return False


def test_derived_build_rebuilds_when_inherited_cc0_source_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run_build_target(monkeypatch, tmp_path, cc0_revision="new") is True


def test_derived_build_skips_when_inherited_cc0_source_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run_build_target(monkeypatch, tmp_path, cc0_revision="old") is False
