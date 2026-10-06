"""Hugging Face datasets 由来の翻訳データ取り込み用アダプタ.

当面の目的:
  - `p1atdev/danbooru-ja-tag-pair-20241015` のような「Danbooruタグ → 日本語」ペアのデータセットを
    builder 側で直接読み込んで `TAG_TRANSLATIONS` に投入できるようにする。

実行環境:
  - オフライン/テストでは `datasets.load_from_disk()` を使えるようにし、
    本番では `datasets.load_dataset(repo_id, ...)` でHFから取得できるようにする。
"""

from __future__ import annotations

import ast
import re
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

from ..core.scripts import guess_language_by_script, has_kana


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


_TRANSLATION_SPLIT_RE = re.compile("[,\uff0c\u3001\uff64\ufe50]")


# 言語ごとの翻訳列候補（先頭ほど優先）。
_TRANSLATION_COLUMNS: dict[str, list[str]] = {
    "ja": ["japanese", "ja", "jp", "translation", "other_names"],
    "zh": ["chinese", "zh", "zh_cn", "zh-cn"],
}
# builder の _extract_translations が言語コードを推定できる列名。
_OUTPUT_COLUMN: dict[str, str] = {"ja": "japanese", "zh": "zh", "ko": "ko"}


def _parse_stringified_list(s: str) -> list[Any] | None:
    """ "['a', 'b']" のような list の文字列表現を list に戻す（失敗時は None）."""
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


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
    # lylogummy/danbooru_wikis_2026 等は other_names を list ではなく文字列で持つ
    parsed = _parse_stringified_list(s)
    if parsed is not None:
        return _explode_translations(parsed)
    # 既存CSVと同様に、カンマ区切りの揺れは複数翻訳として取り込む（重複は許容）
    parts = [p.strip() for p in _TRANSLATION_SPLIT_RE.split(s)]
    cleaned = [p.strip(" \"'\u201c\u201d\u2018\u2019\u300c\u300d") for p in parts]
    return [p for p in cleaned if p]


@dataclass(frozen=True)
class P1atdevDanbooruJaTagPairAdapter:
    """`p1atdev/danbooru-ja-tag-pair-*` 形式の翻訳データを DataFrame 化する."""

    repo_id_or_path: str
    revision: str | None = None
    split: str | None = None
    language: str = "ja"
    # True の場合、訳語を文字種で ja / ko / zh に振り分ける（ラテン文字等は捨てる）。
    # 未フィルタの wiki other_names は多言語が混在するため、生 wiki 系ソース向けに使う。
    classify_scripts: bool = False

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
        jp_col = _pick_column(cols, _TRANSLATION_COLUMNS.get(self.language, [self.language]))
        if tag_col is None or jp_col is None:
            msg = f"Unsupported schema for {self.repo_id_or_path}: columns={cols}"
            raise ValueError(msg)

        records: list[dict[str, str]] = []
        wide_records: list[dict[str, str]] = []
        for row in ds:
            # deleted は翻訳として使わない
            if bool(row.get("is_deleted", False)):
                continue
            tag = str(row.get(tag_col, "")).strip()
            if not tag:
                continue
            translations = _explode_translations(row.get(jp_col))
            for t in translations:
                if self.classify_scripts:
                    lang = guess_language_by_script(t)
                    if lang is None:
                        continue
                    wide_records.append({"source_tag": tag, _OUTPUT_COLUMN.get(lang, lang): t})
                    continue
                # zh 列にアーティスト名などの日本語原表記が入るため、仮名を含む行は捨てる
                if self.language == "zh" and has_kana(t):
                    continue
                # builder の既存翻訳取り込みロジック（_extract_translations）に合わせて言語別列名にする
                records.append({"source_tag": tag, "lang_value": t})

        if self.classify_scripts:
            schema = {"source_tag": pl.Utf8, "japanese": pl.Utf8, "zh": pl.Utf8, "ko": pl.Utf8}
            return pl.DataFrame(wide_records, schema=schema)

        out_col = _OUTPUT_COLUMN.get(self.language, self.language)
        if not records:
            return pl.DataFrame({"source_tag": [], out_col: []})
        return pl.DataFrame(records).rename({"lang_value": out_col})
