from __future__ import annotations

import json
from pathlib import Path

import pytest

import builder_ci.main as ci_main
from builder_ci.manifest import compare_source_revisions


def test_local_bundle_revision_and_change_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ci_main, "_repo_root", lambda: tmp_path)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    source = {"id": "sheet", "kind": "local_bundle", "path": "bundle.json"}
    first = ci_main._fetch_sources([source], tmp_path / "external")
    path.write_text(json.dumps({"version": 2}), encoding="utf-8")
    second = ci_main._fetch_sources([source], tmp_path / "external")
    changes = compare_source_revisions({"sources": first}, second)
    assert len(changes["changed"]) == 1
    assert first[0]["revision"] != second[0]["revision"]
    assert not (tmp_path / "external").exists()


def test_local_bundle_rejects_outside_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ci_main, "_repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="inside"):
        ci_main._local_bundle_path({"path": "../private.json"})


def test_local_bundle_not_treated_as_external_csv(tmp_path: Path) -> None:
    path = ci_main._generate_include_filter(
        [{"kind": "local_bundle", "id": "sheet", "paths_include": ["bundle.json"]}],
        tmp_path / "external",
        tmp_path / "report",
    )
    assert path.read_text(encoding="utf-8") == ""
