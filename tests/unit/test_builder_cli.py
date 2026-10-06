from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from genai_tag_db_dataset_builder import builder


def test_main_passes_hf_translation_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(builder, "build_dataset", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--output",
            str(tmp_path / "out.sqlite"),
            "--hf-ja-translation",
            "owner/wiki",
            "--hf-wiki-multilang",
            "owner/wiki",
            "--hf-zh-translation",
            "owner/zh",
        ],
    )

    builder.main()

    assert captured["hf_ja_translation_datasets"] == ["owner/wiki"]
    assert captured["hf_wiki_multilang_datasets"] == ["owner/wiki"]
    assert captured["hf_zh_translation_datasets"] == ["owner/zh"]
