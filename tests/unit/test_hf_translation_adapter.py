from __future__ import annotations

from pathlib import Path

import polars as pl
from datasets import Dataset

from genai_tag_db_dataset_builder.adapters.hf_translation_adapter import P1atdevDanbooruJaTagPairAdapter


def test_p1atdev_adapter_reads_local_saved_dataset(tmp_path: Path) -> None:
    ds = Dataset.from_dict(
        {
            "tag": ["1girl", "witch"],
            "japanese": ["一人の女の子", "魔女､ウィッチ"],
        }
    )
    save_dir = tmp_path / "hf_ds"
    ds.save_to_disk(save_dir.as_posix())

    df = P1atdevDanbooruJaTagPairAdapter(save_dir.as_posix()).read()
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"source_tag", "japanese"}

    # comma-separated variant is split into multiple rows
    got = {(r["source_tag"], r["japanese"]) for r in df.to_dicts()}
    assert ("1girl", "一人の女の子") in got
    assert ("witch", "魔女") in got
    assert ("witch", "ウィッチ") in got


def test_p1atdev_adapter_supports_title_other_names_schema(tmp_path: Path) -> None:
    ds = Dataset.from_dict(
        {
            "id": [10, 11],
            "title": ["original", "deleted_tag"],
            "other_names": [["オリジナル"], ["消すべき"]],
            "is_deleted": [False, True],
            "type": ["copyright", "general"],
        }
    )
    save_dir = tmp_path / "hf_ds2"
    ds.save_to_disk(save_dir.as_posix())

    df = P1atdevDanbooruJaTagPairAdapter(save_dir.as_posix()).read()
    got = {(r["source_tag"], r["japanese"]) for r in df.to_dicts()}
    assert ("original", "オリジナル") in got
    assert not any(src == "deleted_tag" for (src, _) in got)


def test_adapter_parses_stringified_other_names_and_splits_by_script(tmp_path: Path) -> None:
    # lylogummy/danbooru_wikis_2026 は other_names を list ではなく文字列で持つ
    ds = Dataset.from_dict(
        {
            "title": ["original"],
            "other_names": ["['オリジナル', '原創', '창작', 'oc', 'うちの子']"],
            "is_deleted": [False],
        }
    )
    save_dir = tmp_path / "hf_wiki"
    ds.save_to_disk(save_dir.as_posix())

    plain = P1atdevDanbooruJaTagPairAdapter(save_dir.as_posix()).read()
    assert {r["japanese"] for r in plain.to_dicts()} == {"オリジナル", "原創", "창작", "oc", "うちの子"}

    split = P1atdevDanbooruJaTagPairAdapter(save_dir.as_posix(), classify_scripts=True).read()
    assert set(split.columns) == {"source_tag", "japanese", "zh", "ko"}
    got = {(col, r[col]) for r in split.to_dicts() for col in ("japanese", "zh", "ko") if r[col]}
    assert got == {
        ("japanese", "オリジナル"),
        ("japanese", "うちの子"),
        ("zh", "原創"),
        ("ko", "창작"),
    }


def test_adapter_reads_zh_and_drops_kana_rows(tmp_path: Path) -> None:
    ds = Dataset.from_dict(
        {
            "tag": ["touhou", "dairi"],
            "zh": ["东方Project", "ダイリ"],
            "count": [1, 2],
        }
    )
    save_dir = tmp_path / "hf_zh"
    ds.save_to_disk(save_dir.as_posix())

    df = P1atdevDanbooruJaTagPairAdapter(save_dir.as_posix(), language="zh").read()
    assert set(df.columns) == {"source_tag", "zh"}
    assert [(r["source_tag"], r["zh"]) for r in df.to_dicts()] == [("touhou", "东方Project")]
