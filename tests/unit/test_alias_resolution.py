"""Unit tests for site_tags alias conflict resolution and reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from genai_tag_db_dataset_builder.core.alias_resolution import (
    RESOLUTION_REASON_FIRST_WIN,
    RESOLUTION_REASON_OVERRIDE,
    AliasResolution,
    load_alias_resolution,
    resolve_canonical,
    write_alias_conflicts_tsv,
)


class TestLoadAliasResolution:
    def test_none_path_returns_empty(self) -> None:
        res = load_alias_resolution(None)
        assert res.is_empty()

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        res = load_alias_resolution(tmp_path / "nope.yml")
        assert res.is_empty()

    def test_empty_mapping(self, tmp_path: Path) -> None:
        p = tmp_path / "alias_resolution.yml"
        p.write_text("alias_resolution: {}\n", encoding="utf-8")
        res = load_alias_resolution(p)
        assert res.is_empty()

    def test_valid_override(self, tmp_path: Path) -> None:
        p = tmp_path / "alias_resolution.yml"
        p.write_text(
            "alias_resolution:\n  zerochan.net:\n    アビゲイル: Abigail\n",
            encoding="utf-8",
        )
        res = load_alias_resolution(p)
        assert not res.is_empty()
        assert res.get("zerochan.net", "アビゲイル") == "Abigail"
        assert res.get("zerochan.net", "unknown") is None
        assert res.get("danbooru.donmai.us", "アビゲイル") is None

    def test_invalid_top_structure_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "alias_resolution.yml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_alias_resolution(p)

    def test_invalid_entries_are_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "alias_resolution.yml"
        p.write_text(
            "alias_resolution:\n"
            "  zerochan.net:\n"
            "    good: Abigail\n"
            "    empty: ''\n"
            "    numeric: 123\n"
            "  bad_domain: not_a_mapping\n",
            encoding="utf-8",
        )
        res = load_alias_resolution(p)
        assert res.get("zerochan.net", "good") == "Abigail"
        assert res.get("zerochan.net", "empty") is None
        assert res.get("zerochan.net", "numeric") is None
        assert res.get("bad_domain", "x") is None


class TestResolveCanonical:
    def test_single_candidate_no_conflict(self) -> None:
        selected, conflicts = resolve_canonical(
            "neko_girl", ["cat_girl"], domain="danbooru.donmai.us", resolution=None
        )
        assert selected == "cat_girl"
        assert conflicts == []

    def test_empty_candidates(self) -> None:
        selected, conflicts = resolve_canonical("x", [], domain="d", resolution=None)
        assert selected == ""
        assert conflicts == []

    def test_multi_candidate_first_win(self) -> None:
        selected, conflicts = resolve_canonical(
            "アビゲイル",
            ["Abigail", "Abigail (Gate of Nightmares)"],
            domain="zerochan.net",
            resolution=None,
        )
        assert selected == "Abigail"
        assert len(conflicts) == 1
        assert conflicts[0].rejected_canonical == "Abigail (Gate of Nightmares)"
        assert conflicts[0].resolution_reason == RESOLUTION_REASON_FIRST_WIN

    def test_multi_candidate_override_wins(self) -> None:
        res = AliasResolution({"zerochan.net": {"アビゲイル": "Abigail (Gate of Nightmares)"}})
        selected, conflicts = resolve_canonical(
            "アビゲイル",
            ["Abigail", "Abigail (Gate of Nightmares)"],
            domain="zerochan.net",
            resolution=res,
        )
        assert selected == "Abigail (Gate of Nightmares)"
        assert len(conflicts) == 1
        assert conflicts[0].selected_canonical == "Abigail (Gate of Nightmares)"
        assert conflicts[0].rejected_canonical == "Abigail"
        assert conflicts[0].resolution_reason == RESOLUTION_REASON_OVERRIDE

    def test_override_matching_first_win_produces_no_conflict(self) -> None:
        res = AliasResolution({"zerochan.net": {"アビゲイル": "Abigail"}})
        selected, conflicts = resolve_canonical(
            "アビゲイル", ["Abigail"], domain="zerochan.net", resolution=res
        )
        assert selected == "Abigail"
        assert conflicts == []

    def test_single_candidate_override_differs_records_conflict(self) -> None:
        res = AliasResolution({"zerochan.net": {"アビゲイル": "Abigail"}})
        selected, conflicts = resolve_canonical(
            "アビゲイル", ["Wrong"], domain="zerochan.net", resolution=res
        )
        assert selected == "Abigail"
        assert len(conflicts) == 1
        assert conflicts[0].resolution_reason == RESOLUTION_REASON_OVERRIDE


class TestWriteAliasConflictsTsv:
    def test_no_conflicts_writes_nothing(self, tmp_path: Path) -> None:
        out = write_alias_conflicts_tsv([], tmp_path / "alias_conflicts.tsv")
        assert out is None
        assert not (tmp_path / "alias_conflicts.tsv").exists()

    def test_writes_tsv_with_header(self, tmp_path: Path) -> None:
        _, conflicts = resolve_canonical(
            "アビゲイル",
            ["Abigail", "Abigail (Gate of Nightmares)"],
            domain="zerochan.net",
            resolution=None,
        )
        out = write_alias_conflicts_tsv(conflicts, tmp_path / "report" / "alias_conflicts.tsv")
        assert out is not None
        assert out.exists()
        with open(out, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter="\t"))
        assert rows[0] == [
            "domain",
            "alias",
            "selected_canonical",
            "rejected_canonical",
            "resolution_reason",
        ]
        assert rows[1] == [
            "zerochan.net",
            "アビゲイル",
            "Abigail",
            "Abigail (Gate of Nightmares)",
            "first-win",
        ]
