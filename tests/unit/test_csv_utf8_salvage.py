from __future__ import annotations

from pathlib import Path

import polars as pl

from genai_tag_db_dataset_builder import builder


def test_read_csv_best_effort_skips_invalid_utf8_lines(tmp_path: Path) -> None:
    # ヘッダー無しの想定: source_tag,count
    # 2行目だけ invalid utf-8 を含めて救済できることを確認する
    p = tmp_path / "broken.csv"
    p.write_bytes(b"good,1\nbad,\xff\nok,2\n")

    unknown_dir = tmp_path / "unknown"
    out_dir = tmp_path / "out"

    df = builder._read_csv_best_effort(
        p,
        unknown_report_dir=unknown_dir,
        overrides=None,
        bad_utf8_report_dir=out_dir,
    )

    assert df is not None
    assert "source_tag" in df.columns
    assert "count" in df.columns
    assert df.select(pl.col("source_tag")).to_series().to_list() == ["good", "ok"]

    report_path = out_dir / "csv_invalid_utf8_lines.tsv"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "broken.csv" in report_text
    assert "\t2\t" in report_text  # line_no=2


def test_read_csv_best_effort_headerless_schema_inference(tmp_path: Path) -> None:
    # danbooru 形式の例: tag,type_id,count,deprecated_tags,empty
    p = tmp_path / "danbooru_241016.csv"
    p.write_text("1girl,0,10,sole_female,\n", encoding="utf-8")

    df = builder._read_csv_best_effort(
        p,
        unknown_report_dir=tmp_path / "unknown",
        overrides=None,
        bad_utf8_report_dir=tmp_path / "out",
    )
    assert df is not None
    assert {"source_tag", "type_id", "count", "deprecated_tags"}.issubset(set(df.columns))
