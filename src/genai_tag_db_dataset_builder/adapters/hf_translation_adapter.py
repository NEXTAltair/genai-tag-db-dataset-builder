"""Hugging Face datasets 由来の翻訳データ取り込み用アダプタ.

当面の目的:
  - `p1atdev/danbooru-ja-tag-pair-20241015` のような「Danbooruタグ → 日本語」ペアのデータセットを
    builder 側で直接読み込んで `TAG_TRANSLATIONS` に投入できるようにする。

実行環境:
  - オフライン/テストでは `datasets.load_from_disk()` を使えるようにし、
    本番では `datasets.load_dataset(repo_id, ...)` でHFから取得できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from datasets import (  # type: ignore[import-untyped,unused-ignore]
    Dataset,
    DatasetDict,
    load_dataset,
    load_from_disk,
)


def _is_local_dataset_path(repo_id_or_path: str) -> bool:
    p = Path(repo_id_or_path)
    return p.exists() and p.is_dir()


def _first_split(ds: DatasetDict) -> Dataset:
    # 典型的には "train" のみ。なければ先頭のsplitを選ぶ。
    if "train" in ds:
        return ds["train"]
    return ds[next(iter(ds.keys()))]


def _pick_column(cols: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _explode_translations(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(_explode_translations(v))
        return out
    s = str(value).strip()
    if not s:
        return []
    # 既存CSVと同様に、カンマ区切りの揺れは複数翻訳として取り込む（重複は許容）
    parts = [p.strip() for p in s.split(",")]
    cleaned = [p.strip(" \"'“”‘’「」") for p in parts]
    return [p for p in cleaned if p]


@dataclass(frozen=True)
class P1atdevDanbooruJaTagPairAdapter:
    """`p1atdev/danbooru-ja-tag-pair-*` 形式の翻訳データを DataFrame 化する."""

    repo_id_or_path: str
    revision: str | None = None
    split: str | None = None
    language: str = "ja"

    def read(self) -> pl.DataFrame:
        if _is_local_dataset_path(self.repo_id_or_path):
            ds_obj = load_from_disk(self.repo_id_or_path)
            ds = _first_split(ds_obj) if isinstance(ds_obj, DatasetDict) else ds_obj
        else:
            loaded = load_dataset(self.repo_id_or_path, revision=self.revision)
            ds = _first_split(loaded)
            if self.split:
                # load_dataset で split 指定をしない場合に備えて明示切り替えも許可する
                ds = loaded[self.split]

        cols = list(ds.column_names)

        # フォーマットA（想定していたCSV相当）:
        #   - tag: "1girl"
        #   - japanese: ["猫耳", ...] もしくは "猫耳,ネコミミ"
        #
        # フォーマットB（実データ）:
        #   - title: "original"（=タグ）
        #   - other_names: ["オリジナル", ...]（=日本語名の配列）
        tag_col = _pick_column(cols, ["tag", "source_tag", "danbooru_tag", "title"])
        jp_col = _pick_column(cols, ["japanese", "ja", "jp", "translation", "other_names"])
        if tag_col is None or jp_col is None:
            msg = f"Unsupported schema for {self.repo_id_or_path}: columns={cols}"
            raise ValueError(msg)

        records: list[dict[str, str]] = []
        for row in ds:
            # deleted は翻訳として使わない
            if bool(row.get("is_deleted", False)):
                continue
            tag = str(row.get(tag_col, "")).strip()
            if not tag:
                continue
            translations = _explode_translations(row.get(jp_col))
            for t in translations:
                # builder の既存翻訳取り込みロジック（_extract_translations）に合わせて言語別列名にする
                records.append({"source_tag": tag, "japanese": t})

        if not records:
            return pl.DataFrame({"source_tag": [], "japanese": []})

        return pl.DataFrame(records)
