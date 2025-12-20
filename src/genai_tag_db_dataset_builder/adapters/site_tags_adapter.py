"""deepghs/site_tags 用アダプタ（tags.sqlite）.

deepghs/site_tags は「サイトごとに1つの tags.sqlite」を持ち、サイトごとにスキーマが異なる。
このアダプタは tags.sqlite を読み取り、ビルダーが扱える共通カラムへ正規化する。

設計方針（会話で確定した内容）:
- 18サイトを最初から取り込む前提（初回CC4ビルド）
- TAGS.tag/source_tag は既存 CC0 DB を基本使い回し（足りない分のみ新規追加）
- invalid_tag/bad_tag などへの「吸い込みリダイレクト」は行わない（deprecated として扱う）
- type/category が不明・変換不可の場合は type_id=-1 として投入し、後段のレポート対象
- 翻訳はサイト側に存在する言語を全て取り込む（例: sankaku の trans_*）
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from .base_adapter import BaseAdapter


@dataclass(frozen=True)
class SiteTagsChunk:
    df: pl.DataFrame
    translation_columns: list[str]


class SiteTagsAdapter(BaseAdapter):
    """deepghs/site_tags の tags.sqlite を読むアダプタ."""

    # e621 は tag_aliases に invalid_tag という吸い込み先が存在する（tags テーブルには無い）
    _INVALID_SINKS = {"invalid_tag", "bad_tag"}

    # gelbooru type（文字列）→ type_id（TAG_TYPE_FORMAT_MAPPING の type_id）
    _GELBOORU_TYPE_TO_ID = {
        "general": 0,
        "artist": 1,
        "copyright": 3,
        "character": 4,
        "metadata": 5,
        "deprecated": 6,
    }

    def __init__(
        self,
        sqlite_path: Path | str,
        *,
        format_id: int,
    ) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.format_id = int(format_id)

        if not self.sqlite_path.exists():
            raise FileNotFoundError(f"site_tags sqlite not found: {self.sqlite_path}")

    def read(self) -> pl.DataFrame:
        """小規模ソース向け（全件読み込み）.

        CC4 初回ビルドでは巨大なので、ビルダー側は iter_chunks() を使うこと。
        """
        chunks = [c.df for c in self.iter_chunks(chunk_size=50_000)]
        return pl.concat(chunks) if chunks else pl.DataFrame()

    def validate(self, df: pl.DataFrame) -> bool:
        return "source_tag" in df.columns

    def repair(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def iter_chunks(self, *, chunk_size: int) -> Iterator[SiteTagsChunk]:
        """tags.sqlite を chunk で読み出す（巨大DB向け）."""
        schema = self._detect_schema()

        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            alias_to_tag, deprecated_by_target, invalid_aliases = self._load_alias_mapping(conn, schema)

            # alias として扱うべき source_tag（推奨先があるもの）
            alias_sources = set(alias_to_tag.keys())

            query, translation_columns, translation_select = self._build_select_query(schema)
            cur = conn.execute(query)
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break

                out_rows: list[dict[str, object]] = []
                for r in rows:
                    raw = dict(r)
                    tag_value = raw.get(schema.tag_col)
                    if tag_value is None:
                        continue

                    tag_str = str(tag_value)

                    # alias 元（deprecated 名）として扱うものは、推奨先側の deprecated_tags に寄せるためここではスキップ
                    # （invalid_tag/bad_tag への吸い込みは別扱い）
                    if tag_str in alias_sources:
                        target = alias_to_tag.get(tag_str)
                        if target and target not in self._INVALID_SINKS:
                            continue

                    count_val = raw.get(schema.count_col) if schema.count_col else None
                    type_val = raw.get(schema.type_col) if schema.type_col else None

                    count_i: int | None = None
                    if count_val is not None:
                        try:
                            count_i = int(count_val)
                        except (TypeError, ValueError):
                            count_i = None
                        else:
                            if count_i < 0:
                                count_i = None

                    deprecated_flag = False
                    if schema.deprecated_col:
                        deprecated_flag = bool(raw.get(schema.deprecated_col))
                    if schema.type_is_gelbooru_string and isinstance(type_val, str):
                        deprecated_flag = deprecated_flag or (type_val.lower() == "deprecated")

                    source_created_at = (
                        str(raw.get(schema.created_at_col))
                        if schema.created_at_col and raw.get(schema.created_at_col)
                        else None
                    )
                    source_updated_at = (
                        str(raw.get(schema.updated_at_col))
                        if schema.updated_at_col and raw.get(schema.updated_at_col)
                        else None
                    )

                    out: dict[str, object] = {
                        "source_tag": tag_str,
                        "format_id": self.format_id,
                        "type_id": self._map_type_value(schema, type_val),
                        # count は負値が混入するケースがある（sankaku/konachan 等）。
                        # このDBでは負値は無効として扱い、取り込みをスキップする（None）。
                        "count": count_i,
                        "deprecated_tags": ",".join(deprecated_by_target.get(tag_str, [])),
                        "deprecated": 1 if deprecated_flag else 0,
                        "deprecated_at": None,
                        "source_created_at": source_created_at,
                        "source_updated_at": source_updated_at,
                    }

                    # 翻訳列（存在する言語は全て）
                    if translation_select:
                        for src_col, out_col in translation_select.items():
                            v = raw.get(src_col)
                            if v is None:
                                continue
                            s = str(v).strip()
                            if s:
                                out[out_col] = s

                    out_rows.append(out)

                # NOTE:
                # Polars の dicts→DataFrame 変換は既定で先頭N行だけでスキーマ推定するため、
                # 低頻度の翻訳列（例: zh-CN / de / th など）が後半にしか出ないと列自体が落ちる。
                # ここでは chunk 内の全行を見て推定する。
                df = pl.DataFrame(out_rows, infer_schema_length=None) if out_rows else pl.DataFrame()
                if df.height:
                    yield SiteTagsChunk(df=df, translation_columns=translation_columns)

            # invalid_tag/bad_tag 吸い込み先の alias は「redirect しない」ため、deprecated なタグとして独立投入する
            if invalid_aliases:
                rows = [
                    {
                        "source_tag": alias,
                        "format_id": self.format_id,
                        "type_id": self._invalid_alias_type_id(),
                        "count": None,
                        "deprecated_tags": "",
                        "deprecated": 1,
                        "deprecated_at": None,
                        "source_created_at": None,
                        "source_updated_at": None,
                    }
                    for alias in sorted(invalid_aliases)
                ]
                df_invalid = pl.DataFrame(rows)
                yield SiteTagsChunk(df=df_invalid, translation_columns=[])

        finally:
            conn.close()

    # ----------------------------
    # schema detection / mapping
    # ----------------------------

    @dataclass(frozen=True)
    class _Schema:
        tag_col: str
        count_col: str | None
        type_col: str | None
        deprecated_col: str | None
        created_at_col: str | None
        updated_at_col: str | None
        type_is_gelbooru_string: bool
        translation_map: dict[str, str]  # sqlite col -> language code

    def _detect_schema(self) -> _Schema:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            cols = conn.execute("PRAGMA table_info(tags)").fetchall()
        finally:
            conn.close()

        col_names = [c[1] for c in cols]
        col_types = {c[1]: (c[2] or "").upper() for c in cols}
        s = set(col_names)

        # sankaku: many trans_* columns, plus pool_count/series_count/etc
        if {"name", "post_count", "pool_count", "series_count", "type"}.issubset(s) and any(
            c.startswith("trans_") for c in s
        ):
            translation_map_sankaku = {
                c: c.removeprefix("trans_") for c in col_names if c.startswith("trans_")
            }
            return self._Schema(
                tag_col="name",
                count_col="post_count",
                type_col="type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map=translation_map_sankaku,
            )

        # danbooru-like / safebooru / allthefallen
        if {"name", "post_count", "category"}.issubset(s) and "is_deprecated" in s:
            return self._Schema(
                tag_col="name",
                count_col="post_count",
                type_col="category",
                deprecated_col="is_deprecated",
                created_at_col="created_at" if "created_at" in s else None,
                updated_at_col="updated_at" if "updated_at" in s else None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # e621-like
        if {"name", "post_count", "category"}.issubset(s) and "is_deprecated" not in s:
            return self._Schema(
                tag_col="name",
                count_col="post_count",
                type_col="category",
                deprecated_col=None,
                created_at_col="created_at" if "created_at" in s else None,
                updated_at_col="updated_at" if "updated_at" in s else None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # anime-pictures style (tag_jp/tag_ru)
        if {"tag", "num", "type", "alias"}.issubset(s):
            translation_map_ap: dict[str, str] = {}
            if "tag_jp" in s:
                translation_map_ap["tag_jp"] = "ja"
            if "tag_ru" in s:
                translation_map_ap["tag_ru"] = "ru"
            return self._Schema(
                tag_col="tag",
                count_col="num",
                type_col="type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map=translation_map_ap,
            )

        # pixiv (flags + trans_ja)
        if {"name", "posts", "updated_at"}.issubset(s) and "wiki_url" in s:
            translation_map_pixiv: dict[str, str] = {}
            if "trans_ja" in s:
                translation_map_pixiv["trans_ja"] = "ja"
            return self._Schema(
                tag_col="name",
                count_col="posts",
                type_col=None,
                deprecated_col=None,
                created_at_col=None,
                updated_at_col="updated_at",
                type_is_gelbooru_string=False,
                translation_map=translation_map_pixiv,
            )

        # wallhaven (category_id/category_name)
        if {"name", "posts", "category_id", "category_name"}.issubset(s):
            return self._Schema(
                tag_col="name",
                count_col="posts",
                type_col=None,
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # zerochan
        if {"tag", "type", "total"}.issubset(s):
            return self._Schema(
                tag_col="tag",
                count_col="total",
                type_col=None,
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # gelbooru (type is TEXT, not integer)
        if {"name", "count", "type"}.issubset(s) and col_types.get("type", "") == "TEXT":
            return self._Schema(
                tag_col="name",
                count_col="count",
                type_col="type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=True,
                translation_map={},
            )

        # moebooru-like (type/count/name) with optional ambiguous/id
        if {"name", "count", "type"}.issubset(s) and col_types.get("type", "") in {"INTEGER", "INT"}:
            return self._Schema(
                tag_col="name",
                count_col="count",
                type_col="type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # moebooru variant: (type,count,name) order; used by rule34/hypnohub/etc
        if {"name", "count", "type"}.issubset(s):
            return self._Schema(
                tag_col="name",
                count_col="count",
                type_col="type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        # lolibooru: tag_type + post_count
        if {"name", "post_count", "tag_type"}.issubset(s):
            return self._Schema(
                tag_col="name",
                count_col="post_count",
                type_col="tag_type",
                deprecated_col=None,
                created_at_col=None,
                updated_at_col=None,
                type_is_gelbooru_string=False,
                translation_map={},
            )

        raise ValueError(f"Unsupported site_tags schema: {self.sqlite_path} (cols={col_names})")

    def _build_select_query(self, schema: _Schema) -> tuple[str, list[str], dict[str, str]]:
        """tags テーブルから必要最小限だけ SELECT するクエリを構築する."""

        def _q(name: str) -> str:
            # `trans_zh-CN` のような `-` を含む列名があるため、常に識別子をクォートする。
            # SQLite の識別子クォートは `"`（内部の `"` は `""` でエスケープ）。
            return f'"{name.replace('"', '""')}"'

        cols = [schema.tag_col]
        if schema.count_col:
            cols.append(schema.count_col)
        if schema.type_col:
            cols.append(schema.type_col)
        if schema.deprecated_col:
            cols.append(schema.deprecated_col)
        if schema.created_at_col:
            cols.append(schema.created_at_col)
        if schema.updated_at_col:
            cols.append(schema.updated_at_col)

        translation_select: dict[str, str] = {}
        for src_col, lang in schema.translation_map.items():
            cols.append(src_col)
            translation_select[src_col] = self._normalize_language_code(lang)

        # 重複を除去しつつ順序維持
        seen: set[str] = set()
        select_cols: list[str] = []
        for c in cols:
            if c in seen:
                continue
            seen.add(c)
            select_cols.append(c)

        sql = f"SELECT {', '.join(_q(c) for c in select_cols)} FROM tags"
        translation_columns = sorted(set(translation_select.values()))
        return sql, translation_columns, translation_select

    def _normalize_language_code(self, code: str) -> str:
        # source 側の trans_zh-CN 等はそのまま保持（case も維持）
        return str(code).strip()

    def _invalid_alias_type_id(self) -> int:
        # e621: invalid tag category = 6
        if self.format_id == 2:
            return 6
        return -1

    def _map_type_value(self, schema: _Schema, value: object) -> int:
        if value is None or schema.type_col is None:
            return -1
        if schema.type_is_gelbooru_string:
            s = str(value).strip().lower()
            return self._GELBOORU_TYPE_TO_ID.get(s, -1)
        if isinstance(value, int | float | str | bytes):
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return -1

    def _load_alias_mapping(
        self, conn: sqlite3.Connection, schema: _Schema
    ) -> tuple[dict[str, str], dict[str, list[str]], set[str]]:
        """alias 情報をロードして deprecated_tags へ変換する.

        Returns:
            alias_to_tag: alias_source_tag -> preferred_source_tag
            deprecated_by_target: preferred_source_tag -> list[alias_source_tag]
            invalid_aliases: invalid_tag/bad_tag への alias_source_tag の集合（redirectしない）
        """
        alias_to_tag: dict[str, str] = {}
        deprecated_by_target: dict[str, list[str]] = {}
        invalid_aliases: set[str] = set()

        # 1) tag_aliases テーブルがある場合（danbooru/e621/moebooru/zerochan等）
        has_tag_aliases = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tag_aliases'"
        ).fetchone()
        if has_tag_aliases:
            rows = conn.execute("SELECT alias, tag FROM tag_aliases").fetchall()
            for alias, tag in rows:
                if alias is None or tag is None:
                    continue
                a = str(alias)
                t = str(tag)

                # 重複 alias は先勝ち + レポート（count が取れない場合の方針）
                if a in alias_to_tag and alias_to_tag[a] != t:
                    logger.warning(
                        f"[site_tags] alias conflict (first-win): {a!r} -> {alias_to_tag[a]!r} / {t!r} "
                        f"({self.sqlite_path})"
                    )
                    continue
                alias_to_tag[a] = t

                if t in self._INVALID_SINKS:
                    invalid_aliases.add(a)
                    continue
                deprecated_by_target.setdefault(t, []).append(a)

            for k in list(deprecated_by_target.keys()):
                deprecated_by_target[k] = sorted(set(deprecated_by_target[k]))
            return alias_to_tag, deprecated_by_target, invalid_aliases

        # 2) tags.alias が「推奨先 id」を指す形式（anime-pictures）
        if (
            schema.tag_col == "tag"
            and conn.execute("SELECT 1 FROM pragma_table_info('tags') WHERE name='alias'").fetchone()
        ):
            # alias は tags.id を参照する
            rows = conn.execute(
                """
                SELECT a.tag AS alias, t.tag AS tag
                FROM tags a
                JOIN tags t ON a.alias = t.id
                WHERE a.alias IS NOT NULL
                """
            ).fetchall()
            for alias, tag in rows:
                if alias is None or tag is None:
                    continue
                a = str(alias)
                t = str(tag)
                if a in alias_to_tag and alias_to_tag[a] != t:
                    logger.warning(
                        f"[site_tags] alias conflict (first-win): {a!r} -> {alias_to_tag[a]!r} / {t!r} "
                        f"({self.sqlite_path})"
                    )
                    continue
                alias_to_tag[a] = t
                deprecated_by_target.setdefault(t, []).append(a)

            for k in list(deprecated_by_target.keys()):
                deprecated_by_target[k] = sorted(set(deprecated_by_target[k]))
            return alias_to_tag, deprecated_by_target, invalid_aliases

        return alias_to_tag, deprecated_by_target, invalid_aliases
