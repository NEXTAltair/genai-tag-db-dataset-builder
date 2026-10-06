from __future__ import annotations

from pathlib import Path

from builder_ci.readme import generate_readme

SOURCES_YML = """
sources:
  - id: ame
    kind: hf_dataset
    repo_id: ame-la/danbooru-tags-data-zh
    url: https://huggingface.co/datasets/ame-la/danbooru-tags-data-zh
    license: mit
    copyright: "Copyright (c) 2026 amenorira"
    applies_to: [mit]
    enabled: true
"""

EFFECTS = (
    "source\taction\trows_read\tdb_changes\tnote\n"
    "hf://datasets/ame-la/danbooru-tags-data-zh\ttags_created\t10\t5\tdanbooru_tag_list\n"
    "hf://datasets/ame-la/danbooru-tags-data-zh\timported\t10\t8\thf_zh_translation\n"
)


def test_readme_lists_license_copyright_and_dedupes_source(tmp_path: Path) -> None:
    sources_yml = tmp_path / "sources.yml"
    sources_yml.write_text(SOURCES_YML, encoding="utf-8")
    report = tmp_path / "out" / "report"
    report.mkdir(parents=True)
    (report / "source_effects.tsv").write_text(EFFECTS, encoding="utf-8")

    text = generate_readme("mit", tmp_path / "out", sources_yml, {"repo_id": "cc0_local", "path": "x"})

    lines = [ln for ln in text.splitlines() if ln.startswith("- ame (")]
    assert len(lines) == 1  # tags_created と imported の 2 行が 1 行にまとまる
    assert "(mit)" in lines[0]
    assert "Copyright (c) 2026 amenorira" in lines[0]
