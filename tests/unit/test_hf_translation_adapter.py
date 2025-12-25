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
