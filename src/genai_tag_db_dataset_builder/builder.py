"""データセットビルダー（オーケストレーター）.

ローカルにある複数ソース（site_tags など）を統合し、配布用の SQLite DB を生成する。
入力の揺れ（tag 列の意味差・括弧エスケープ・大文字小文字など）は adapter/core 側の正規化に寄せ、
ここでは「取り込み順（再現性）」「衝突の検出とレポート」「DB 作成の一連」を担う。
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import polars as pl

from genai_tag_db_dataset_builder.adapters.hf_translation_adapter import P1atdevDanbooruJaTagPairAdapter
from genai_tag_db_dataset_builder.adapters.site_tags_adapter import SiteTagsAdapter
from genai_tag_db_dataset_builder.core.database import (
    apply_connection_pragmas,
    build_indexes,
    create_database,
    optimize_database,
)
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data
from genai_tag_db_dataset_builder.core.merge import merge_tags, normalize_tag, process_deprecated_tags
from genai_tag_db_dataset_builder.tools.migrate_db import migrate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ConflictRow:
    source: str
    tag_id: int
    format_id: int
    kind: str
    existing: str
    incoming: str


def _chunked(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _extract_all_tags_from_deprecated(df: pl.DataFrame) -> set[str]:
    """source_tagとdeprecated_tagsから全てのタグを抽出.

    Two-pass処理の第1パスで使用。deprecated_tags列に含まれるaliasも含めて
    全タグを事前に抽出し、TAGS登録漏れを防ぐ。

    Args:
        df: source_tag列とdeprecated_tags列を含むDataFrame

    Returns:
        抽出された全タグのset（重複除去済み）
    """
    tags: list[str] = []

    # source_tag列から抽出
    if "source_tag" in df.columns:
        tags.extend([str(v) for v in df["source_tag"].to_list() if v is not None])

    # deprecated_tags列から抽出
    if "deprecated_tags" in df.columns:
        for deprecated_str in df["deprecated_tags"].to_list():
            if deprecated_str and isinstance(deprecated_str, str):
                for t in deprecated_str.split(","):
                    t = t.strip()
                    if t:
                        tags.append(t)

    return set(tags)


def _clean_translation_part(text: str) -> str:
    return text.strip().strip(" \"'“”‘’「」")


def _extract_translations(
    df: pl.DataFrame,
    tags_mapping: dict[str, int],
) -> list[tuple[int, str, str]]:
    """翻訳データを抽出する.

    Args:
        df: 翻訳ソースDataFrame（tag列 + 言語別列を含む）
        tags_mapping: tag名 → tag_idのマッピング

    Returns:
        (tag_id, language, translation)のtupleリスト
    """
    results: list[tuple[int, str, str]] = []

    def _split_translation_values(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for v in value:
                out.extend(_split_translation_values(v))
            return out
        text = str(value).strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(",")]
        cleaned = [_clean_translation_part(p) for p in parts]
        return [p for p in cleaned if p]

    # tag列から正規化タグ名を取得
    if "tag" not in df.columns and "source_tag" in df.columns:
        tag_col = "source_tag"
    elif "tag" in df.columns:
        tag_col = "tag"
    else:
        return results

    # 言語列を検出（tag, source_tag以外の列）
    lang_columns = [col for col in df.columns if col not in {tag_col, "source_tag", "tag"}]

    for row in df.to_dicts():
        tag_value = row.get(tag_col)
        if not tag_value:
            continue

        # 正規化してtag_idを取得
        normalized_tag = normalize_tag(str(tag_value))
        tag_id = tags_mapping.get(normalized_tag)
        if tag_id is None:
            continue

        # 各言語列から翻訳を取得
        for lang_col in lang_columns:
            translation = row.get(lang_col)
            translations = _split_translation_values(translation)
            if translations:
                # 言語コードを推定（japanese → ja等）
                language = _infer_language_code(lang_col)
                for item in translations:
                    results.append((tag_id, language, item))

    return results


def _infer_language_code(column_name: str) -> str:
    """列名から言語コードを推定する.

    Args:
        column_name: 列名（例: "japanese", "english", "ja", "en"）

    Returns:
        言語コード（例: "ja", "en"）
    """
    col_lower = column_name.lower()

    # deepghs/site_tags 系: trans_zh-CN / trans_zh-HK など（列名がそのまま言語コードを含む）
    if col_lower.startswith("trans_"):
        # 元の大小をなるべく維持する（zh-CN など）
        return column_name[len("trans_") :]

    # deepghs/site_tags 系: tag_jp / tag_ru のような suffix 形式
    if col_lower.endswith("_jp"):
        return "ja"
    if col_lower.endswith("_ja"):
        return "ja"
    if col_lower.endswith("_ru"):
        return "ru"
    if col_lower.endswith("_ko"):
        return "ko"
    if col_lower.endswith("_zh"):
        return "zh"

    # 直接言語コードの場合
    if col_lower in {
        "ja",
        "en",
        "zh",
        "ko",
        "fr",
        "de",
        "es",
        "ru",
        "pt",
        "it",
        "vi",
        "th",
        "zh-cn",
        "zh-hk",
    }:
        return col_lower

    # 言語名からコードに変換
    lang_map = {
        "japanese": "ja",
        "english": "en",
        "chinese": "zh",
        "korean": "ko",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru",
    }

    return lang_map.get(col_lower, col_lower[:2])  # デフォルトは先頭2文字


def _split_translation_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(_split_translation_values(v))
        return out
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    cleaned = [_clean_translation_part(p) for p in parts]
    return [p for p in cleaned if p]


def _normalize_language_value(value: str) -> str:
    """DB側の language 値を正規化する（例: japanese -> ja）。"""
    v = str(value).strip()
    if not v:
        return v
    return _infer_language_code(v)


_SITE_TAGS_SITE_TO_FORMAT_ID: dict[str, int] = {
    "anime-pictures.net": 14,
    "booru.allthefallen.moe": 5,
    "chan.sankakucomplex.com": 6,  # sankaku
    "danbooru.donmai.us": 1,
    "e621.net": 2,
    "en.pixiv.net": 18,  # pixiv
    "gelbooru.com": 7,
    "hypnohub.net": 10,
    "konachan.com": 11,
    "konachan.net": 12,
    "lolibooru.moe": 13,
    "pixiv.net": 18,
    "rule34.xxx": 8,
    "safebooru.donmai.us": 4,
    "wallhaven.cc": 15,
    "xbooru.com": 9,
    "yande.re": 16,
    "zerochan.net": 17,
}


def _infer_site_tags_format_id(site_dir_name: str) -> int:
    return _SITE_TAGS_SITE_TO_FORMAT_ID.get(site_dir_name, 0)


def _load_source_filters(path: Path | None) -> tuple[set[str], list[str]]:
    """ソースフィルタ（include/exclude）ファイルを読む.

    - 1行1エントリ（相対パス推奨）
    - 空行 / # から始まる行は無視
    - ワイルドカード（*, ?, []）を含む行はパターンとして扱う
    """
    if path is None:
        return set(), []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Source filter file not found: {p}")

    exact: set[str] = set()
    patterns: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if any(ch in line for ch in "*?[]"):
            patterns.append(line)
        else:
            exact.add(line)
    return exact, patterns


def _match_source(name: str, *, exact: set[str], patterns: list[str]) -> bool:
    if name in exact:
        return True
    return any(fnmatch(name, pat) for pat in patterns)


def _should_include_source(
    source_name: str,
    *,
    include_exact: set[str],
    include_patterns: list[str],
    exclude_exact: set[str],
    exclude_patterns: list[str],
) -> bool:
    """source_name を取り込むかどうかを判定する."""
    if (include_exact or include_patterns) and not _match_source(
        source_name, exact=include_exact, patterns=include_patterns
    ):
        return False
    return not (
        (exclude_exact or exclude_patterns)
        and _match_source(source_name, exact=exclude_exact, patterns=exclude_patterns)
    )


def _infer_source_timestamp_utc_midnight(path: Path) -> str | None:
    """ファイル名から日付を推定し、UTCの 00:00:00 に固定したタイムスタンプ文字列を返す.

    対応例:
      - danbooru_241016.csv -> 2024-10-16 00:00:00+00:00
      - danbooru_20241016.csv -> 2024-10-16 00:00:00+00:00
    """
    name = path.name.lower()
    m8 = re.search(r"_(\d{8})(?:\D|$)", name)
    if m8:
        ymd = m8.group(1)
        yyyy, mm, dd = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
        return f"{yyyy:04d}-{mm:02d}-{dd:02d} 00:00:00+00:00"

    m6 = re.search(r"_(\d{6})(?:\D|$)", name)
    if m6:
        ymd = m6.group(1)
        yy, mm, dd = int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])
        yyyy = 2000 + yy
        return f"{yyyy:04d}-{mm:02d}-{dd:02d} 00:00:00+00:00"

    return None


def _is_authoritative_count_source(path: Path) -> bool:
    """最新スナップショットとして count を上書きするソースかどうか."""
    name = path.name.lower()
    return bool(re.search(r"^danbooru_\d{6,8}\.csv$", name))


def _replace_usage_counts_for_format(
    conn: sqlite3.Connection,
    *,
    format_id: int,
    counts_by_tag_id: dict[int, int],
    timestamp: str,
) -> None:
    """TAG_USAGE_COUNTS の特定 format_id をスナップショットで置換する.

    注意: これは TAGS/TAG_STATUS を削除しない。削除するのは TAG_USAGE_COUNTS の当該 format_id 行のみ。
    """
    conn.execute("DELETE FROM TAG_USAGE_COUNTS WHERE format_id = ?", (format_id,))
    rows = [(tag_id, format_id, count, timestamp, timestamp) for tag_id, count in counts_by_tag_id.items()]
    conn.executemany(
        "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str | bytes):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _ensure_temp_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TEMP TABLE IF NOT EXISTS TMP_TAG_STATUS_INPUT (
            tag_id INTEGER NOT NULL,
            format_id INTEGER NOT NULL,
            type_id INTEGER NOT NULL,
            alias INTEGER NOT NULL,
            preferred_tag_id INTEGER NOT NULL
        );
        CREATE TEMP TABLE IF NOT EXISTS TMP_TAG_USAGE_INPUT (
            tag_id INTEGER NOT NULL,
            format_id INTEGER NOT NULL,
            count INTEGER NOT NULL
        );
        """
    )


def _insert_tags(conn: sqlite3.Connection, tags_df: pl.DataFrame) -> None:
    rows = list(
        zip(
            tags_df["tag_id"].to_list(),
            tags_df["source_tag"].to_list(),
            tags_df["tag"].to_list(),
            strict=False,
        )
    )
    for chunk in _chunked(rows, 10_000):
        conn.executemany(
            "INSERT INTO TAGS (tag_id, source_tag, tag) VALUES (?, ?, ?)",
            chunk,
        )


def _insert_tag_status_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, int, int, int, int]],
    *,
    source_name: str,
    conflicts: list[_ConflictRow],
) -> None:
    # 同一 (tag_id, format_id) が同一バッチ内に重複し得る（同じタグが複数行に現れる等）。
    # SQLite 側の NOT EXISTS だけだと、同一 INSERT 文内の重複を防げず UNIQUE 制約で落ちるので、
    # 事前にここで畳み込む（差分がある場合は衝突としてレポート）。
    dedup: dict[tuple[int, int], tuple[int, int, int]] = {}
    for tag_id, format_id, type_id, alias, preferred_tag_id in rows:
        key = (tag_id, format_id)
        current = (type_id, alias, preferred_tag_id)
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = current
            continue

        if prev != current:
            conflicts.append(
                _ConflictRow(
                    source=source_name,
                    tag_id=tag_id,
                    format_id=format_id,
                    kind="TAG_STATUS_INPUT_DUPLICATE",
                    existing=f"type_id={prev[0]},alias={prev[1]},preferred_tag_id={prev[2]}",
                    incoming=f"type_id={current[0]},alias={current[1]},preferred_tag_id={current[2]}",
                )
            )

        # 選択ルール: canonical(=alias=0) を優先。次点で preferred_tag_id が小さい方。
        if prev[1] == 0:
            continue
        if current[1] == 0:
            dedup[key] = current
            continue
        if current[2] < prev[2]:
            dedup[key] = current

    rows = [(k[0], k[1], v[0], v[1], v[2]) for k, v in dedup.items()]

    _ensure_temp_tables(conn)
    conn.execute("DELETE FROM TMP_TAG_STATUS_INPUT;")
    conn.executemany(
        "INSERT INTO TMP_TAG_STATUS_INPUT (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    conflict_cur = conn.execute(
        """
        SELECT
            t.tag_id,
            t.format_id,
            s.type_id AS type_id_existing,
            t.type_id AS type_id_incoming,
            s.alias AS alias_existing,
            t.alias AS alias_incoming,
            s.preferred_tag_id AS preferred_tag_id_existing,
            t.preferred_tag_id AS preferred_tag_id_incoming
        FROM TMP_TAG_STATUS_INPUT t
        JOIN TAG_STATUS s
          ON s.tag_id = t.tag_id AND s.format_id = t.format_id
        WHERE
          s.type_id != t.type_id
          OR s.alias != t.alias
          OR s.preferred_tag_id != t.preferred_tag_id;
        """
    )
    for (
        tag_id,
        format_id,
        type_id_existing,
        type_id_incoming,
        alias_existing,
        alias_incoming,
        preferred_existing,
        preferred_incoming,
    ) in conflict_cur.fetchall():
        conflicts.append(
            _ConflictRow(
                source=source_name,
                tag_id=int(tag_id),
                format_id=int(format_id),
                kind="TAG_STATUS",
                existing=f"type_id={type_id_existing},alias={alias_existing},preferred_tag_id={preferred_existing}",
                incoming=f"type_id={type_id_incoming},alias={alias_incoming},preferred_tag_id={preferred_incoming}",
            )
        )

    conn.execute(
        """
        INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id)
        SELECT t.tag_id, t.format_id, t.type_id, t.alias, t.preferred_tag_id
        FROM TMP_TAG_STATUS_INPUT t
        WHERE NOT EXISTS (
          SELECT 1 FROM TAG_STATUS s
          WHERE s.tag_id = t.tag_id AND s.format_id = t.format_id
        );
        """
    )


def _insert_usage_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, int, int]],
    *,
    source_name: str,
    conflicts: list[_ConflictRow],
) -> None:
    if not rows:
        return

    _ensure_temp_tables(conn)
    conn.execute("DELETE FROM TMP_TAG_USAGE_INPUT;")
    conn.executemany(
        "INSERT INTO TMP_TAG_USAGE_INPUT (tag_id, format_id, count) VALUES (?, ?, ?)",
        rows,
    )

    # 同一 (tag_id, format_id) の count は「最大値」を採用する。
    # - posting site/集計元が違うと count の意味が変わり得るため（小さい方が学習データ枚数などの可能性）
    # - 手動判断が不要な衝突として扱う
    #
    # まず入力側で MAX 集約し、DB 側は「UPDATE → INSERT」の2段で max(existing, incoming) を実現する。
    conn.executescript(
        """
        CREATE TEMP TABLE IF NOT EXISTS TMP_TAG_USAGE_AGG (
            tag_id INTEGER NOT NULL,
            format_id INTEGER NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY (tag_id, format_id)
        );
        DELETE FROM TMP_TAG_USAGE_AGG;
        INSERT INTO TMP_TAG_USAGE_AGG (tag_id, format_id, count)
        SELECT tag_id, format_id, MAX(count) AS count
        FROM TMP_TAG_USAGE_INPUT
        GROUP BY tag_id, format_id;
        """
    )

    # 既存行はより大きい方に更新
    conn.execute(
        """
        UPDATE TAG_USAGE_COUNTS
        SET count = CASE
          WHEN (
            SELECT a.count
            FROM TMP_TAG_USAGE_AGG a
            WHERE a.tag_id = TAG_USAGE_COUNTS.tag_id
              AND a.format_id = TAG_USAGE_COUNTS.format_id
          ) > TAG_USAGE_COUNTS.count
          THEN (
            SELECT a.count
            FROM TMP_TAG_USAGE_AGG a
            WHERE a.tag_id = TAG_USAGE_COUNTS.tag_id
              AND a.format_id = TAG_USAGE_COUNTS.format_id
          )
          ELSE TAG_USAGE_COUNTS.count
        END
        WHERE EXISTS (
          SELECT 1
          FROM TMP_TAG_USAGE_AGG a
          WHERE a.tag_id = TAG_USAGE_COUNTS.tag_id
            AND a.format_id = TAG_USAGE_COUNTS.format_id
        );
        """
    )

    # 未存在行は挿入
    conn.execute(
        """
        INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count)
        SELECT a.tag_id, a.format_id, a.count
        FROM TMP_TAG_USAGE_AGG a
        WHERE NOT EXISTS (
          SELECT 1 FROM TAG_USAGE_COUNTS u
          WHERE u.tag_id = a.tag_id AND u.format_id = a.format_id
        );
        """
    )

    # conflicts は残しているが、TAG_USAGE_COUNTS の差分は自動解決のためレポートしない。
    _ = source_name
    _ = conflicts


def _insert_translations(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, str]],
) -> None:
    for chunk in _chunked(rows, 10_000):
        conn.executemany(
            "INSERT OR IGNORE INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
            chunk,
        )


def _delete_ja_translations_by_value_list(
    conn: sqlite3.Connection,
    *,
    values: list[str],
) -> int:
    """TAG_TRANSLATIONS から language='ja' の特定 translation を削除する.

    Returns:
        削除された行数（SQLiteの total_changes 差分）
    """
    if not values:
        return 0

    conn.execute("DROP TABLE IF EXISTS temp._cleanup_values")
    conn.execute("CREATE TEMP TABLE _cleanup_values (value TEXT PRIMARY KEY)")

    # NOTE: 巨大リストでも効率よく削除できるよう、TEMP表に詰めてJOINで消す
    for chunk in _chunked(values, 10_000):
        conn.executemany(
            "INSERT OR IGNORE INTO _cleanup_values (value) VALUES (?)",
            [(v,) for v in chunk if v],
        )

    conn.execute(
        """
        DELETE FROM TAG_TRANSLATIONS
        WHERE language = 'ja'
          AND translation IN (SELECT value FROM _cleanup_values)
        """
    )
    deleted = int(conn.execute("SELECT changes()").fetchone()[0])
    conn.execute("DROP TABLE IF EXISTS temp._cleanup_values")
    conn.commit()
    return deleted


def _delete_translations_ascii_only_for_languages(
    conn: sqlite3.Connection,
    *,
    languages: set[str],
) -> int:
    """指定 language のうち、ASCIIのみの translation を削除する."""
    if not languages:
        return 0

    q_marks = ",".join(["?"] * len(languages))
    rows = conn.execute(
        f"SELECT translation_id, translation FROM TAG_TRANSLATIONS WHERE language IN ({q_marks})",
        tuple(sorted(languages)),
    ).fetchall()
    if not rows:
        return 0

    bad_ids: list[int] = []
    for translation_id, text in rows:
        if not text:
            continue
        s = str(text)
        if all(ord(ch) < 128 for ch in s):
            bad_ids.append(int(translation_id))

    if not bad_ids:
        return 0

    changes_before = conn.total_changes
    for chunk in _chunked(bad_ids, 900):  # SQLiteの変数制限回避
        marks = ",".join(["?"] * len(chunk))
        conn.execute(
            f"DELETE FROM TAG_TRANSLATIONS WHERE translation_id IN ({marks})",
            tuple(chunk),
        )
    conn.commit()
    return int(conn.total_changes - changes_before)


def _delete_translations_missing_required_script(
    conn: sqlite3.Connection,
    *,
    language: str,
) -> int:
    """languageごとの必須文字種を含まない translation を削除する.

    方針（ユーザー決定）:
    - ja: ひらがな/カタカナ/漢字（CJK）のいずれも含まないものは誤りとして削除
    - zh: 漢字（CJK）を含まないものは誤りとして削除
    - ko: ハングルを含まないものは誤りとして削除
    """
    import re

    lang = language
    if lang == "ja":
        required = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    elif lang == "zh":
        required = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    elif lang == "ko":
        required = re.compile(r"[\uac00-\ud7af]")
    else:
        return 0

    rows = conn.execute(
        "SELECT translation_id, translation FROM TAG_TRANSLATIONS WHERE language = ?",
        (lang,),
    ).fetchall()
    if not rows:
        return 0

    bad_ids: list[int] = []
    for translation_id, text in rows:
        if not text:
            continue
        s = str(text)
        if not required.search(s):
            bad_ids.append(int(translation_id))

    if not bad_ids:
        return 0

    changes_before = conn.total_changes
    for chunk in _chunked(bad_ids, 900):
        marks = ",".join(["?"] * len(chunk))
        conn.execute(
            f"DELETE FROM TAG_TRANSLATIONS WHERE translation_id IN ({marks})",
            tuple(chunk),
        )
    conn.commit()
    return int(conn.total_changes - changes_before)


def _split_comma_delimited_translations(conn: sqlite3.Connection) -> int:
    """TAG_TRANSLATIONS のカンマ区切り翻訳を分割して再投入する."""
    rows = conn.execute(
        "SELECT translation_id, tag_id, language, translation, created_at, updated_at "
        "FROM TAG_TRANSLATIONS WHERE translation LIKE '%,%'"
    ).fetchall()
    if not rows:
        return 0

    deleted_ids: list[int] = []
    inserts: list[tuple[int, str, str, str | None, str | None]] = []
    for translation_id, tag_id, language, translation, created_at, updated_at in rows:
        parts = _split_translation_values(translation)
        if len(parts) <= 1:
            continue
        deleted_ids.append(int(translation_id))
        for part in parts:
            inserts.append((int(tag_id), str(language), part, created_at, updated_at))

    if not deleted_ids:
        return 0

    changes_before = conn.total_changes
    for chunk in _chunked(deleted_ids, 900):
        marks = ",".join(["?"] * len(chunk))
        conn.execute(
            f"DELETE FROM TAG_TRANSLATIONS WHERE translation_id IN ({marks})",
            tuple(chunk),
        )
    for tag_id, language, translation, created_at, updated_at in inserts:
        conn.execute(
            "INSERT OR IGNORE INTO TAG_TRANSLATIONS (tag_id, language, translation, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (tag_id, language, translation, created_at, updated_at),
        )
    conn.commit()
    return int(conn.total_changes - changes_before)


def _write_conflicts_tsv(path: Path, conflicts: list[_ConflictRow]) -> None:
    if not conflicts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "kind", "tag_id", "format_id", "existing", "incoming"],
            delimiter="\t",
        )
        writer.writeheader()
        for c in conflicts:
            writer.writerow(
                {
                    "source": c.source,
                    "kind": c.kind,
                    "tag_id": c.tag_id,
                    "format_id": c.format_id,
                    "existing": c.existing,
                    "incoming": c.incoming,
                }
            )


def _write_skipped_tsv(path: Path, skipped: list[tuple[str, str]]) -> None:
    if not skipped:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "reason"], delimiter="\t")
        writer.writeheader()
        for src, reason in skipped:
            writer.writerow({"source": src, "reason": reason})


def _write_source_effects_tsv(path: Path, effects: list[dict[str, object]]) -> None:
    if not effects:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "action", "rows_read", "db_changes", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        for r in effects:
            writer.writerow(
                {
                    "source": r.get("source", ""),
                    "action": r.get("action", ""),
                    "rows_read": r.get("rows_read", 0),
                    "db_changes": r.get("db_changes", 0),
                    "note": r.get("note", ""),
                }
            )


_TAG_STATUS_UPSERT_IF_CHANGED_SQL = """
INSERT INTO TAG_STATUS (
  tag_id,
  format_id,
  type_id,
  alias,
  preferred_tag_id,
  deprecated,
  deprecated_at,
  source_created_at
)
VALUES (?, ?, ?, ?, ?, COALESCE(?, 0), ?, ?)
ON CONFLICT(tag_id, format_id) DO UPDATE SET
  type_id = excluded.type_id,
  alias = excluded.alias,
  preferred_tag_id = excluded.preferred_tag_id,
  deprecated = excluded.deprecated,
  deprecated_at = excluded.deprecated_at,
  source_created_at = excluded.source_created_at,
  updated_at = CURRENT_TIMESTAMP
WHERE
  TAG_STATUS.type_id IS NOT excluded.type_id
  OR TAG_STATUS.alias IS NOT excluded.alias
  OR TAG_STATUS.preferred_tag_id IS NOT excluded.preferred_tag_id
  OR TAG_STATUS.deprecated IS NOT excluded.deprecated
  OR TAG_STATUS.deprecated_at IS NOT excluded.deprecated_at
  OR TAG_STATUS.source_created_at IS NOT excluded.source_created_at;
""".strip()

_TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL = """
INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count, created_at, updated_at)
VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), COALESCE(?, CURRENT_TIMESTAMP))
ON CONFLICT(tag_id, format_id) DO UPDATE SET
  count = excluded.count,
  created_at = excluded.created_at,
  updated_at = excluded.updated_at
WHERE
  excluded.count > TAG_USAGE_COUNTS.count;
""".strip()


def _repair_placeholder_tag_ids(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """placeholder の __missing_tag_id_*__ を既存の tag_id に付け替える。

    方針:
    - placeholder には元の文字列が無いため、TAG_STATUS の preferred_tag_id に寄せる以外の自動復元は不可。
    - (preferred_tag_id, format_id) の TAG_STATUS が既に存在する場合は、placeholder 側の TAG_STATUS は削除する。
    - preferred_tag_id == tag_id かつ alias=true のような破損行は削除する。
    - 参照が無くなった placeholder TAGS 行は削除する。

    Returns:
        実行内容のレポート行（TSV出力用）。
    """
    repairs: list[dict[str, object]] = []

    placeholder_rows = conn.execute(
        "SELECT tag_id, tag FROM TAGS WHERE tag LIKE '__missing_tag_id_%__'"
    ).fetchall()
    if not placeholder_rows:
        return repairs

    for missing_id, placeholder_tag in placeholder_rows:
        status_rows = conn.execute(
            "SELECT format_id, type_id, alias, preferred_tag_id FROM TAG_STATUS WHERE tag_id = ?",
            (missing_id,),
        ).fetchall()

        if not status_rows:
            conn.execute("DELETE FROM TAGS WHERE tag_id = ?", (missing_id,))
            repairs.append(
                {
                    "missing_id": int(missing_id),
                    "action": "delete_placeholder_only",
                    "format_id": "",
                    "preferred_tag_id": "",
                    "note": "No TAG_STATUS rows",
                    "placeholder_tag": placeholder_tag,
                }
            )
            continue

        status_map: dict[int, int] = {}
        for format_id, type_id, _alias, preferred_tag_id in status_rows:
            if int(preferred_tag_id) != int(missing_id):
                status_map[int(format_id)] = int(preferred_tag_id)
            if int(preferred_tag_id) == int(missing_id):
                conn.execute(
                    "DELETE FROM TAG_STATUS WHERE tag_id = ? AND format_id = ?",
                    (missing_id, format_id),
                )
                repairs.append(
                    {
                        "missing_id": int(missing_id),
                        "action": "delete_invalid_self_preferred",
                        "format_id": int(format_id),
                        "preferred_tag_id": int(preferred_tag_id),
                        "note": "preferred_tag_id==tag_id (invalid for alias row)",
                        "placeholder_tag": placeholder_tag,
                    }
                )
                continue

            exists = conn.execute(
                "SELECT 1 FROM TAG_STATUS WHERE tag_id = ? AND format_id = ?",
                (preferred_tag_id, format_id),
            ).fetchone()
            if exists:
                conn.execute(
                    "DELETE FROM TAG_STATUS WHERE tag_id = ? AND format_id = ?",
                    (missing_id, format_id),
                )
                repairs.append(
                    {
                        "missing_id": int(missing_id),
                        "action": "delete_redundant_status",
                        "format_id": int(format_id),
                        "preferred_tag_id": int(preferred_tag_id),
                        "note": "Preferred tag already has TAG_STATUS for this format",
                        "placeholder_tag": placeholder_tag,
                    }
                )
            else:
                conn.execute(
                    "UPDATE TAG_STATUS SET tag_id = ?, alias = 0, preferred_tag_id = ? "
                    "WHERE tag_id = ? AND format_id = ?",
                    (preferred_tag_id, preferred_tag_id, missing_id, format_id),
                )
                repairs.append(
                    {
                        "missing_id": int(missing_id),
                        "action": "reassign_status_to_preferred",
                        "format_id": int(format_id),
                        "preferred_tag_id": int(preferred_tag_id),
                        "note": f"Adopted placeholder TAG_STATUS as canonical (type_id={int(type_id)})",
                        "placeholder_tag": placeholder_tag,
                    }
                )

        # TAG_USAGE_COUNTS は format_id があるので、TAG_STATUS の preferred_tag_id に寄せられる
        usage_rows = conn.execute(
            "SELECT format_id, count FROM TAG_USAGE_COUNTS WHERE tag_id = ?",
            (missing_id,),
        ).fetchall()
        for format_id, count in usage_rows:
            preferred = status_map.get(int(format_id))
            if preferred is None or preferred == int(missing_id):
                conn.execute(
                    "DELETE FROM TAG_USAGE_COUNTS WHERE tag_id = ? AND format_id = ?",
                    (missing_id, format_id),
                )
                repairs.append(
                    {
                        "missing_id": int(missing_id),
                        "action": "delete_usage_count_unmappable",
                        "format_id": int(format_id),
                        "preferred_tag_id": preferred if preferred is not None else "",
                        "note": "No usable preferred_tag_id for this format",
                        "placeholder_tag": placeholder_tag,
                    }
                )
                continue

            conn.execute(
                "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count) VALUES (?, ?, ?) "
                "ON CONFLICT(tag_id, format_id) DO UPDATE SET count = MAX(count, excluded.count)",
                (preferred, int(format_id), int(count)),
            )
            conn.execute(
                "DELETE FROM TAG_USAGE_COUNTS WHERE tag_id = ? AND format_id = ?",
                (missing_id, format_id),
            )
            repairs.append(
                {
                    "missing_id": int(missing_id),
                    "action": "move_usage_count_to_preferred",
                    "format_id": int(format_id),
                    "preferred_tag_id": int(preferred),
                    "note": "Moved TAG_USAGE_COUNTS row to preferred tag_id",
                    "placeholder_tag": placeholder_tag,
                }
            )

        extra_translation = conn.execute(
            "SELECT COUNT(*) FROM TAG_TRANSLATIONS WHERE tag_id = ?",
            (missing_id,),
        ).fetchone()[0]
        if extra_translation:
            repairs.append(
                {
                    "missing_id": int(missing_id),
                    "action": "has_extra_rows_needs_manual",
                    "format_id": "",
                    "preferred_tag_id": "",
                    "note": f"TAG_TRANSLATIONS={int(extra_translation)}",
                    "placeholder_tag": placeholder_tag,
                }
            )

        refs = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM TAG_STATUS WHERE tag_id = ? OR preferred_tag_id = ?) +
                (SELECT COUNT(*) FROM TAG_TRANSLATIONS WHERE tag_id = ?) +
                (SELECT COUNT(*) FROM TAG_USAGE_COUNTS WHERE tag_id = ?)
            """,
            (missing_id, missing_id, missing_id, missing_id),
        ).fetchone()[0]
        if int(refs) == 0:
            conn.execute("DELETE FROM TAGS WHERE tag_id = ?", (missing_id,))
            repairs.append(
                {
                    "missing_id": int(missing_id),
                    "action": "delete_placeholder_after_repair",
                    "format_id": "",
                    "preferred_tag_id": "",
                    "note": "No remaining references",
                    "placeholder_tag": placeholder_tag,
                }
            )

    return repairs


def _write_placeholder_repairs_tsv(report_dir: Path, repairs: list[dict[str, object]]) -> None:
    if not repairs:
        return

    out_path = report_dir / "placeholder_tag_id_repairs.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["missing_id", "action", "format_id", "preferred_tag_id", "note", "placeholder_tag"]
        )
        for r in repairs:
            writer.writerow(
                [
                    r["missing_id"],
                    r["action"],
                    r["format_id"],
                    r["preferred_tag_id"],
                    r["note"],
                    r["placeholder_tag"],
                ]
            )
    logger.warning(f"Placeholder tag_id repairs report: {out_path} ({len(repairs)} rows)")


def _strip_category_prefix(tag: str) -> str | None:
    """カテゴリprefix（meta:/artist:）を除去してベースタグ名を返す。"""
    if tag.startswith("meta:"):
        return tag[len("meta:") :].strip()
    if tag.startswith("artist:"):
        return tag[len("artist:") :].strip()
    return None


def _repair_non_e621_prefix_preferred(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """非e621(format_id!=2)で、preferredがカテゴリprefix側になっている行を修正する。

    既存DB由来のデータで、danbooru/derpibooru/unknown 側に `meta:`/`artist:` が混入することがある。
    方針としてカテゴリprefixは「保持しつつ alias 解決で prefix 無し側へ寄せる」ため、非e621では
    - preferred_tag が prefix なら prefix 無しタグへ寄せる
    - tag 自体が prefix の場合も prefix 無しタグへ寄せる（alias=true）
    を行う。

    Returns:
        修正内容のレポート行（TSV出力用）
    """
    repairs: list[dict[str, object]] = []

    # 候補: 非e621で、tag または preferred_tag に prefix が含まれる行
    rows = conn.execute(
        """
        SELECT
            ts.tag_id,
            ts.format_id,
            ts.type_id,
            ts.alias,
            ts.preferred_tag_id,
            t.tag AS tag,
            tp.tag AS preferred_tag
        FROM TAG_STATUS ts
        JOIN TAGS t ON t.tag_id = ts.tag_id
        JOIN TAGS tp ON tp.tag_id = ts.preferred_tag_id
        WHERE
            ts.format_id != 2
            AND (
                t.tag LIKE 'meta:%' OR t.tag LIKE 'artist:%'
                OR tp.tag LIKE 'meta:%' OR tp.tag LIKE 'artist:%'
            )
        """
    ).fetchall()

    for tag_id, format_id, _type_id, alias, preferred_tag_id, tag, preferred_tag in rows:
        base = _strip_category_prefix(preferred_tag) or _strip_category_prefix(tag)
        if not base:
            continue

        base_row = conn.execute("SELECT tag_id FROM TAGS WHERE tag = ? LIMIT 1", (base,)).fetchone()
        if base_row is None:
            repairs.append(
                {
                    "format_id": int(format_id),
                    "tag_id": int(tag_id),
                    "tag": tag,
                    "old_preferred_tag_id": int(preferred_tag_id),
                    "old_preferred_tag": preferred_tag,
                    "new_preferred_tag_id": "",
                    "new_preferred_tag": "",
                    "new_alias": "",
                    "reason": "base_tag_not_found",
                }
            )
            continue

        base_id = int(base_row[0])
        if base_id == int(tag_id):
            new_alias = 0
            new_preferred = int(tag_id)
        else:
            new_alias = 1
            new_preferred = base_id

        if int(preferred_tag_id) == new_preferred and int(alias) == new_alias:
            continue

        conn.execute(
            "UPDATE TAG_STATUS SET alias = ?, preferred_tag_id = ? WHERE tag_id = ? AND format_id = ?",
            (new_alias, new_preferred, tag_id, format_id),
        )
        new_preferred_tag = conn.execute(
            "SELECT tag FROM TAGS WHERE tag_id = ?", (new_preferred,)
        ).fetchone()[0]
        repairs.append(
            {
                "format_id": int(format_id),
                "tag_id": int(tag_id),
                "tag": tag,
                "old_preferred_tag_id": int(preferred_tag_id),
                "old_preferred_tag": preferred_tag,
                "new_preferred_tag_id": int(new_preferred),
                "new_preferred_tag": new_preferred_tag,
                "new_alias": int(new_alias),
                "reason": "prefix_to_base",
            }
        )

    return repairs


def _write_non_e621_prefix_repairs_tsv(report_dir: Path, repairs: list[dict[str, object]]) -> None:
    if not repairs:
        return
    out_path = report_dir / "non_e621_prefix_preferred_repairs.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "format_id",
                "tag_id",
                "tag",
                "old_preferred_tag_id",
                "old_preferred_tag",
                "new_preferred_tag_id",
                "new_preferred_tag",
                "new_alias",
                "reason",
            ]
        )
        for r in repairs:
            writer.writerow(
                [
                    r["format_id"],
                    r["tag_id"],
                    r["tag"],
                    r["old_preferred_tag_id"],
                    r["old_preferred_tag"],
                    r["new_preferred_tag_id"],
                    r["new_preferred_tag"],
                    r["new_alias"],
                    r["reason"],
                ]
            )
    logger.warning(f"Non-e621 prefix preferred repairs report: {out_path} ({len(repairs)} rows)")


def _write_tag_status_conflicts_tsv(
    report_dir: Path, tag_status_df: pl.DataFrame, tags_df: pl.DataFrame
) -> None:
    """TAG_STATUS の衝突（同一(tag, format_id)で属性が複数パターン）を詳細TSVで出力する。"""
    report_dir.mkdir(parents=True, exist_ok=True)

    format_name_map = {0: "unknown", 1: "danbooru", 2: "e621", 3: "derpibooru"}

    merged = tag_status_df.join(tags_df.select(["tag_id", "tag"]), on="tag_id", how="left")
    preferred_tags = tags_df.select(
        pl.col("tag_id").alias("preferred_tag_id"),
        pl.col("tag").alias("preferred_tag"),
    )
    merged = merged.join(preferred_tags, on="preferred_tag_id", how="left")
    merged = merged.with_columns(
        [
            (pl.col("tag").is_null() | (pl.col("tag") == "")).alias("tag_missing"),
            pl.when(pl.col("tag").is_null() | (pl.col("tag") == ""))
            .then(pl.format("__missing_tag_id_{}__", pl.col("tag_id")))
            .otherwise(pl.col("tag"))
            .alias("tag_key"),
        ]
    )

    conflicts = (
        merged.group_by(["tag_key", "format_id"])
        .agg(
            [
                pl.col("alias").n_unique().alias("alias_variants"),
                pl.col("type_id").n_unique().alias("type_id_variants"),
                pl.col("preferred_tag_id").n_unique().alias("preferred_tag_id_variants"),
                pl.len().alias("rows"),
            ]
        )
        .filter(
            (pl.col("alias_variants") > 1)
            | (pl.col("type_id_variants") > 1)
            | (pl.col("preferred_tag_id_variants") > 1)
        )
    )

    out_path = report_dir / "tag_status_conflicts.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "format_id",
                "format_name",
                "tag",
                "tag_missing",
                "tag_id",
                "alias",
                "type_id",
                "preferred_tag_id",
                "preferred_tag",
            ]
        )

    out_summary = report_dir / "tag_status_conflicts_summary.tsv"
    with open(out_summary, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "format_id",
                "format_name",
                "tag",
                "rows",
                "alias_variants",
                "type_id_variants",
                "preferred_tag_id_variants",
            ]
        )

    if conflicts.is_empty():
        logger.warning(f"TAG_STATUS conflicts report: {out_path} (0 rows), summary: {out_summary} (0 keys)")
        return

    details = (
        merged.join(conflicts.select(["tag_key", "format_id"]), on=["tag_key", "format_id"], how="inner")
        .with_columns(
            pl.col("format_id")
            .map_elements(lambda x: format_name_map.get(int(x), "unknown"), return_dtype=pl.String)
            .alias("format_name")
        )
        .select(
            [
                "format_id",
                "format_name",
                "tag_key",
                "tag_missing",
                "tag_id",
                "alias",
                "type_id",
                "preferred_tag_id",
                "preferred_tag",
            ]
        )
        .sort(["format_id", "tag_key", "tag_id"])
    )

    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for r in details.to_dicts():
            writer.writerow(
                [
                    r["format_id"],
                    r["format_name"],
                    r["tag_key"],
                    r["tag_missing"],
                    r["tag_id"],
                    r["alias"],
                    r["type_id"],
                    r["preferred_tag_id"],
                    r["preferred_tag"],
                ]
            )

    summary = (
        conflicts.with_columns(
            pl.col("format_id")
            .map_elements(lambda x: format_name_map.get(int(x), "unknown"), return_dtype=pl.String)
            .alias("format_name")
        )
        .select(
            [
                "format_id",
                "format_name",
                "tag_key",
                "rows",
                "alias_variants",
                "type_id_variants",
                "preferred_tag_id_variants",
            ]
        )
        .sort(["format_id", "tag_key"])
    )

    with open(out_summary, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for r in summary.to_dicts():
            writer.writerow(
                [
                    r["format_id"],
                    r["format_name"],
                    r["tag_key"],
                    r["rows"],
                    r["alias_variants"],
                    r["type_id_variants"],
                    r["preferred_tag_id_variants"],
                ]
            )

    logger.warning(
        f"TAG_STATUS conflicts report: {out_path} ({len(details)} rows), summary: {out_summary} ({len(summary)} keys)"
    )


def _repair_missing_type_format_mapping(
    conn: sqlite3.Connection, *, report_dir: Path | None = None
) -> None:
    """TAG_STATUS の (format_id, type_id) が TAG_TYPE_FORMAT_MAPPING に存在しない場合を救済する.

    外部キー制約の整合性を満たすために、未知の組み合わせを TAG_TYPE_FORMAT_MAPPING に追加する。
    - type_id が TAG_TYPE_NAME に存在する（0..16）場合は「同じID」を type_name_id として採用
    - それ以外は unknown(0) にフォールバック
    """
    cur = conn.execute("SELECT MAX(type_name_id) FROM TAG_TYPE_NAME")
    max_type_name_id = cur.fetchone()[0] or 0

    missing = conn.execute(
        """
        SELECT
            ts.format_id,
            ts.type_id,
            COUNT(*) AS rows
        FROM TAG_STATUS ts
        LEFT JOIN TAG_TYPE_FORMAT_MAPPING m
            ON m.format_id = ts.format_id
            AND m.type_id = ts.type_id
        WHERE m.format_id IS NULL
        GROUP BY ts.format_id, ts.type_id
        ORDER BY rows DESC, ts.format_id, ts.type_id
        """
    ).fetchall()

    if not missing:
        return

    repairs: list[tuple[int, int, int, int, str]] = []
    for format_id, type_id, rows in missing:
        if 0 <= int(type_id) <= int(max_type_name_id):
            type_name_id = int(type_id)
            reason = "type_id_as_type_name_id"
        else:
            type_name_id = 0
            reason = "fallback_unknown"

        conn.execute(
            "INSERT OR IGNORE INTO TAG_TYPE_FORMAT_MAPPING (format_id, type_id, type_name_id, description) VALUES (?, ?, ?, ?)",
            (int(format_id), int(type_id), int(type_name_id), f"auto-added ({reason})"),
        )
        repairs.append((int(format_id), int(type_id), int(type_name_id), int(rows), reason))

    conn.commit()

    if report_dir is None:
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "missing_type_format_mapping_repairs.tsv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["format_id", "type_id", "type_name_id", "rows", "reason"])
        writer.writerows(repairs)
    logger.warning(f"Missing type/format mapping repairs report: {out_path} ({len(repairs)} pairs)")


def _export_to_parquet(
    db_path: Path,
    output_dir: Path,
    *,
    tables: list[str] | None = None,
) -> list[Path]:
    """SQLiteデータベースから指定テーブルをParquet形式で出力する.

    HuggingFace Dataset Viewerでの閲覧を可能にするため、主要テーブルをParquet形式で出力します。

    Args:
        db_path: SQLiteデータベースのパス
        output_dir: Parquetファイルの出力ディレクトリ
        tables: 出力するテーブル名のリスト（Noneの場合はデフォルトテーブルを出力）

    Returns:
        出力されたParquetファイルのパスのリスト
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # デフォルトテーブル（主要データテーブル）
    default_tables = [
        "TAGS",
        "TAG_STATUS",
        "TAG_TRANSLATIONS",
        "TAG_USAGE_COUNTS",
        "TAG_FORMATS",
        "TAG_TYPE_NAME",
        "TAG_TYPE_FORMAT_MAPPING",
    ]

    tables_to_export = tables if tables is not None else default_tables

    logger.info(f"Exporting {len(tables_to_export)} tables to Parquet: {output_dir}")

    exported_files: list[Path] = []

    def _read_query(q: str) -> pl.DataFrame:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(q)
            rows = cur.fetchall()
            if not rows:
                return pl.DataFrame()
            return pl.DataFrame([dict(r) for r in rows])
        finally:
            conn.close()

    for table_name in tables_to_export:
        try:
            df = _read_query(f"SELECT * FROM {table_name}")

            # Parquetファイルとして出力
            output_path = output_dir / f"{table_name.lower()}.parquet"
            df.write_parquet(output_path)

            logger.info(
                f"Exported {table_name}: {len(df)} rows → {output_path.name} "
                f"({output_path.stat().st_size / 1024 / 1024:.2f} MB)"
            )
            exported_files.append(output_path)

        except Exception as e:
            logger.warning(f"Failed to export table {table_name}: {e}")
            continue

    logger.info(f"Export complete: {len(exported_files)} Parquet files created")
    return exported_files


def _export_danbooru_view_parquet(
    db_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = 50_000,
) -> list[Path]:
    """HF Dataset Viewer向けのdanbooruビュー（format_id=1）をParquetで出力する.

    方針（決定済み）:
    - まずは danbooru のみ（format_id=1）
    - 1行=そのformatにおける推奨タグ（preferred_tag_id）
    - `deprecated_tags`: alias=1 の逆引き（TAGS.tag の list[str]）
    - 翻訳列は `lang_ja`, `lang_zh` 固定（list[str]集約）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sep = "\u001f"  # group_concat 用の区切り（タグに出にくい）

    conn = sqlite3.connect(db_path)
    try:
        min_id, max_id = conn.execute(
            "SELECT MIN(preferred_tag_id), MAX(preferred_tag_id) FROM TAG_STATUS WHERE format_id = 1"
        ).fetchone()
    finally:
        conn.close()

    if min_id is None or max_id is None:
        logger.warning("[Parquet] No TAG_STATUS rows for format_id=1, skipping danbooru parquet export")
        return []

    exported: list[Path] = []

    base_query = f"""
    WITH canon AS (
      SELECT DISTINCT preferred_tag_id AS tag_id
      FROM TAG_STATUS
      WHERE format_id = 1
        AND preferred_tag_id BETWEEN {{lo}} AND {{hi}}
    ),
    canon_status AS (
      SELECT preferred_tag_id AS tag_id, type_id
      FROM TAG_STATUS
      WHERE format_id = 1
        AND tag_id = preferred_tag_id
        AND preferred_tag_id BETWEEN {{lo}} AND {{hi}}
    ),
    alias_rev AS (
      SELECT preferred_tag_id AS tag_id,
             group_concat(tag, '{sep}') AS deprecated_tags_str
      FROM (
        SELECT ts.preferred_tag_id, a.tag
        FROM TAG_STATUS ts
        JOIN TAGS a ON a.tag_id = ts.tag_id
        WHERE ts.format_id = 1
          AND ts.alias = 1
          AND ts.preferred_tag_id BETWEEN {{lo}} AND {{hi}}
        ORDER BY ts.preferred_tag_id, ts.tag_id
      )
      GROUP BY preferred_tag_id
    ),
    tr_ja AS (
      SELECT tag_id,
             group_concat(translation, '{sep}') AS lang_ja_str
      FROM (
        SELECT tag_id, translation
        FROM TAG_TRANSLATIONS
        WHERE language = 'ja'
          AND tag_id BETWEEN {{lo}} AND {{hi}}
        ORDER BY tag_id, translation_id
      )
      GROUP BY tag_id
    ),
    tr_zh AS (
      SELECT tag_id,
             group_concat(translation, '{sep}') AS lang_zh_str
      FROM (
        SELECT tag_id, translation
        FROM TAG_TRANSLATIONS
        WHERE language = 'zh'
          AND tag_id BETWEEN {{lo}} AND {{hi}}
        ORDER BY tag_id, translation_id
      )
      GROUP BY tag_id
    )
    SELECT
      t.tag_id AS tag_id,
      t.tag AS tag,
      f.format_name AS format_name,
      COALESCE(tn.type_name, 'unknown') AS type_name,
      uc.count AS count,
      alias_rev.deprecated_tags_str AS deprecated_tags_str,
      tr_ja.lang_ja_str AS lang_ja_str,
      tr_zh.lang_zh_str AS lang_zh_str
    FROM canon
    JOIN TAGS t ON t.tag_id = canon.tag_id
    JOIN TAG_FORMATS f ON f.format_id = 1
    LEFT JOIN canon_status cs ON cs.tag_id = t.tag_id
    LEFT JOIN TAG_TYPE_FORMAT_MAPPING m
      ON m.format_id = 1 AND m.type_id = cs.type_id
    LEFT JOIN TAG_TYPE_NAME tn
      ON tn.type_name_id = m.type_name_id
    LEFT JOIN TAG_USAGE_COUNTS uc
      ON uc.tag_id = t.tag_id AND uc.format_id = 1
    LEFT JOIN alias_rev
      ON alias_rev.tag_id = t.tag_id
    LEFT JOIN tr_ja
      ON tr_ja.tag_id = t.tag_id
    LEFT JOIN tr_zh
      ON tr_zh.tag_id = t.tag_id
    ORDER BY t.tag_id
    """

    shard = 0
    start = int(min_id)
    end_max = int(max_id)
    logger.info(
        f"[Parquet] Exporting danbooru view (format_id=1) to {output_dir} "
        f"(preferred_tag_id range: {start}..{end_max}, chunk={chunk_size})"
    )

    for lo in range(start, end_max + 1, chunk_size):
        hi = min(lo + chunk_size - 1, end_max)
        try:
            query = base_query.format(lo=int(lo), hi=int(hi))
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(query)
                rows = cur.fetchall()
            finally:
                conn.close()
            if not rows:
                continue
            df = pl.DataFrame(
                [dict(r) for r in rows],
                schema={
                    "tag_id": pl.Int64,
                    "tag": pl.Utf8,
                    "format_name": pl.Utf8,
                    "type_name": pl.Utf8,
                    "count": pl.Int64,
                    "deprecated_tags_str": pl.Utf8,
                    "lang_ja_str": pl.Utf8,
                    "lang_zh_str": pl.Utf8,
                },
            )
        except Exception as e:
            logger.warning(f"[Parquet] Failed to read danbooru chunk {lo}-{hi}: {e}")
            continue

        def _split_list(col: str) -> pl.Expr:
            return (
                pl.col(col)
                .fill_null("")
                .str.split(sep)
                .list.eval(pl.element().filter(pl.element() != ""))
                .list.eval(pl.element().str.split(","))
                .list.flatten()
                .list.eval(pl.element().str.strip_chars())
                .list.eval(pl.element().filter(pl.element() != ""))
                .alias(col.replace("_str", ""))
            )

        df = df.with_columns(
            [
                _split_list("deprecated_tags_str"),
                _split_list("lang_ja_str"),
                _split_list("lang_zh_str"),
            ]
        ).drop(["deprecated_tags_str", "lang_ja_str", "lang_zh_str"])

        out_path = output_dir / f"danbooru-{shard:05d}.parquet"
        df.write_parquet(out_path, compression="zstd")
        exported.append(out_path)
        shard += 1

    logger.info(f"[Parquet] Exported danbooru parquet shards: {len(exported)} files")
    return exported


def build_dataset(
    output_path: Path | str,
    sources_dir: Path | str,
    version: str,
    report_dir: Path | str | None = None,
    include_sources_path: Path | str | None = None,
    exclude_sources_path: Path | str | None = None,
    hf_ja_translation_datasets: list[str] | None = None,
    parquet_output_dir: Path | str | None = None,
    base_db_path: Path | str | None = None,
    overwrite: bool = False,
) -> None:
    """データセットをビルドして配布用DBを生成.

    Args:
        output_path: 出力DBファイルのパス
        sources_dir: ソースディレクトリ（site_tags などを含む）
        version: データセットバージョン（例: "v4.1.0"）
        report_dir: レポート出力先ディレクトリ（Noneの場合はレポート出力なし）
        hf_ja_translation_datasets: Hugging Face datasets から日本語翻訳を取り込む（例: p1atdev/danbooru-ja-tag-pair-20241015）
        parquet_output_dir: Parquet出力先ディレクトリ（Noneの場合はParquet出力なし）
        base_db_path: ベースとなる既存SQLiteファイル（MIT版ビルド等で使用）。指定時はPhase 0/1をスキップ
        overwrite: 既存のoutput_pathを上書きするか

    Raises:
        FileExistsError: output_pathが既に存在し、overwrite=False の場合
        FileNotFoundError: sources_dirが存在しない場合、またはbase_db_pathが存在しない場合
    """
    output_path = Path(output_path)
    sources_dir = Path(sources_dir)

    # base_db_path が指定されている場合はチェック
    if base_db_path:
        base_db_path = Path(base_db_path)
        if not base_db_path.exists():
            msg = f"Base DB not found: {base_db_path}"
            raise FileNotFoundError(msg)

    if output_path.exists() and not overwrite:
        msg = f"Output DB already exists: {output_path}. Use --overwrite to replace."
        raise FileExistsError(msg)

    if not sources_dir.exists():
        msg = f"Sources directory not found: {sources_dir}"
        raise FileNotFoundError(msg)

    # レポート出力ディレクトリ
    report_dir_path = Path(report_dir) if report_dir else None
    if report_dir_path:
        report_dir_path.mkdir(parents=True, exist_ok=True)

    include_exact, include_patterns = (
        _load_source_filters(Path(include_sources_path)) if include_sources_path else (set(), [])
    )
    exclude_exact, exclude_patterns = (
        _load_source_filters(Path(exclude_sources_path)) if exclude_sources_path else (set(), [])
    )

    # スキップされたソースのレポート（ソース名, 理由）
    skipped: list[tuple[str, str]] = []
    # ソースごとの「実際にDBへ影響があったか」を後で判定できるように、DB変更量を記録する。
    # 目的: MIT版READMEに「影響したMITソースだけ」を列挙するため。
    source_effects: list[dict[str, object]] = []

    placeholder_repairs: list[dict[str, object]] = []

    # base_db_path が指定されている場合は、Phase 0/1 をスキップして既存DBをコピー
    skip_phase_0_1 = base_db_path is not None
    if skip_phase_0_1:
        import shutil

        assert base_db_path is not None
        logger.info(f"[Base DB] Copying base database from {base_db_path} to {output_path}")
        if output_path.exists():
            output_path.unlink()
        shutil.copy2(base_db_path, output_path)
        logger.info("[Base DB] Applying migrations to copied database.")
        migrate(output_path)
        logger.info("[Base DB] Base database copied. Skipping Phase 0/1.")

    # Phase 0: DB 作成・マスターデータ登録（base_db_path 指定時はスキップ）
    if not skip_phase_0_1:
        logger.info(f"[Phase 0] Creating database: {output_path}")
        if output_path.exists():
            output_path.unlink()
        create_database(output_path)

        logger.info("[Phase 0] Inserting master data (FORMATS, TYPES, LANGUAGES)")
        initialize_master_data(output_path)

    conn = sqlite3.connect(output_path)
    apply_connection_pragmas(conn, profile="build")
    existing_tags: set[str] = set()
    tags_mapping: dict[str, int] = {}
    try:
        # Phase 1: tags_v4.db 取り込みは廃止。base_db_path 未指定の場合は空から開始する。
        if not skip_phase_0_1:
            logger.info("[Phase 1] Skipped tags_v4.db import; starting from empty database.")
            next_tag_id = 1
            existing_tags = set()
            tags_mapping = {}
        else:
            # Phase 1 スキップ: 既存DBからtags_mapping等を読み取り
            logger.info("[Phase 1] Skipped (base_db_path specified). Loading existing tags from base DB.")
            cursor = conn.execute("SELECT tag FROM TAGS")
            existing_tags = {row[0] for row in cursor.fetchall()}
            cursor = conn.execute("SELECT MAX(tag_id) FROM TAGS")
            max_tag_id = cursor.fetchone()[0]
            next_tag_id = (max_tag_id or 0) + 1
            cursor = conn.execute("SELECT tag_id, tag FROM TAGS")
            tags_mapping = {row[1]: row[0] for row in cursor.fetchall()}
            source_effects.append(
                {
                    "source": "base_db",
                    "action": "loaded",
                    "rows_read": len(existing_tags),
                    "db_changes": 0,
                    "note": "loaded from base database",
                }
            )

        # Phase 1.5: Hugging Face datasets から翻訳（日本語）を取り込む（任意）
        if hf_ja_translation_datasets:
            logger.info(
                f"[Phase 1.5] Importing HF JA translations: {len(hf_ja_translation_datasets)} dataset(s)"
            )
            for repo_id in hf_ja_translation_datasets:
                source_name = f"hf://datasets/{repo_id}"
                changes_before = conn.total_changes
                try:
                    df_hf = P1atdevDanbooruJaTagPairAdapter(repo_id).read()
                except Exception as e:
                    logger.warning(f"[Phase 1.5] Failed to load translations from {source_name}: {e}")
                    source_effects.append(
                        {
                            "source": source_name,
                            "action": "read_failed",
                            "rows_read": 0,
                            "db_changes": 0,
                            "note": str(e),
                        }
                    )
                    continue

                trans_rows = _extract_translations(df_hf, tags_mapping)
                if not trans_rows:
                    logger.warning(
                        f"[Phase 1.5] No translations extracted from {source_name} (tags not found?)"
                    )
                    continue

                for tag_id, language, translation in trans_rows:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
                            (tag_id, language, translation),
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to insert translation: tag_id={tag_id}, lang={language}, trans={translation} ({e})"
                        )
                conn.commit()
                source_effects.append(
                    {
                        "source": source_name,
                        "action": "imported",
                        "rows_read": int(len(trans_rows)),
                        "db_changes": int(conn.total_changes - changes_before),
                        "note": "hf_ja_translation",
                    }
                )
                logger.info(f"[Phase 1.5] Imported translations: {source_name} (rows={len(trans_rows)})")

        # Phase 2: deepghs/site_tags (CC-BY-4.0) 統合
        site_tags_root = sources_dir / "external_sources" / "site_tags"
        if site_tags_root.exists() and site_tags_root.is_dir():
            sqlite_files = sorted(site_tags_root.glob("*/tags.sqlite"))
            if sqlite_files:
                logger.info(
                    f"[Phase 2] Importing deepghs/site_tags: {len(sqlite_files)} sqlite(s) under {site_tags_root}"
                )

            for sqlite_path in sqlite_files:
                source_name = sqlite_path.relative_to(sources_dir).as_posix()
                if not _should_include_source(
                    source_name,
                    include_exact=include_exact,
                    include_patterns=include_patterns,
                    exclude_exact=exclude_exact,
                    exclude_patterns=exclude_patterns,
                ):
                    skipped.append((source_name, "filtered"))
                    source_effects.append(
                        {
                            "source": source_name,
                            "action": "filtered",
                            "rows_read": 0,
                            "db_changes": 0,
                            "note": "source filter",
                        }
                    )
                    continue

                format_id = _infer_site_tags_format_id(sqlite_path.parent.name)
                if format_id == 0:
                    skipped.append((source_name, "unknown_site_format"))
                    source_effects.append(
                        {
                            "source": source_name,
                            "action": "validation_failed",
                            "rows_read": 0,
                            "db_changes": 0,
                            "note": f"unknown site dir: {sqlite_path.parent.name}",
                        }
                    )
                    continue

                changes_before = conn.total_changes
                rows_read_total = 0
                translation_rows_total = 0

                site_tags_adapter = SiteTagsAdapter(sqlite_path, format_id=format_id)
                for site_chunk in site_tags_adapter.iter_chunks(chunk_size=50_000):
                    df = site_chunk.df
                    rows_read_total += int(df.height)

                    if "source_tag" not in df.columns:
                        continue

                    # 必須/補助列の付与（欠損は既定値）
                    df = df.with_columns(pl.lit(format_id).cast(pl.Int64).alias("format_id"))
                    if "type_id" not in df.columns:
                        df = df.with_columns(pl.lit(-1).cast(pl.Int64).alias("type_id"))
                    else:
                        df = df.with_columns(
                            pl.col("type_id").cast(pl.Int64, strict=False).fill_null(-1).alias("type_id")
                        )
                    if "deprecated_tags" not in df.columns:
                        df = df.with_columns(pl.lit("").alias("deprecated_tags"))
                    else:
                        df = df.with_columns(
                            pl.coalesce([pl.col("deprecated_tags"), pl.lit("")]).alias("deprecated_tags")
                        )
                    if "deprecated" not in df.columns:
                        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("deprecated"))
                    else:
                        df = df.with_columns(
                            pl.col("deprecated")
                            .cast(pl.Int64, strict=False)
                            .fill_null(0)
                            .alias("deprecated")
                        )
                    if "deprecated_at" not in df.columns:
                        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("deprecated_at"))
                    if "source_created_at" not in df.columns:
                        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("source_created_at"))
                    if "source_updated_at" not in df.columns:
                        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("source_updated_at"))

                    # 1) TAGS 登録（source_tag + deprecated_tags）
                    candidates = list(_extract_all_tags_from_deprecated(df))
                    if candidates:
                        new_tags_df = merge_tags(
                            existing_tags, pl.DataFrame({"source_tag": candidates}), next_tag_id
                        )
                        if len(new_tags_df) > 0:
                            for row in new_tags_df.to_dicts():
                                conn.execute(
                                    "INSERT INTO TAGS (tag_id, tag, source_tag) VALUES (?, ?, ?)",
                                    (row["tag_id"], row["tag"], row["source_tag"]),
                                )
                            conn.commit()
                            added_tags = new_tags_df["tag"].to_list()
                            existing_tags.update(added_tags)
                    tags_mapping.update(
                        dict(
                            zip(
                                new_tags_df["tag"].to_list(),
                                new_tags_df["tag_id"].to_list(),
                                strict=False,
                            )
                        )
                    )
                    if len(new_tags_df) > 0:
                        max_tag_id = _safe_int(new_tags_df["tag_id"].max())
                        if max_tag_id is not None:
                            next_tag_id = max_tag_id + 1

                    # 2) TAG_STATUS / TAG_USAGE / TAG_TRANSLATIONS
                    st_rows_site: list[tuple[int, int, int, int, int, int, str | None, str | None]] = []
                    usage_rows_site: list[tuple[int, int, int, str | None]] = []

                    source_tags = df["source_tag"].to_list()
                    deprecated_list = df["deprecated_tags"].to_list()
                    format_ids = df["format_id"].to_list()
                    type_ids = df["type_id"].to_list()
                    deprecated_flags = df["deprecated"].to_list()
                    created_ats = df["source_created_at"].to_list()
                    updated_ats = df["source_updated_at"].to_list()
                    counts = df["count"].to_list() if "count" in df.columns else [None] * len(df)

                    for (
                        raw_source_tag,
                        dep,
                        fmt,
                        tid,
                        dep_flag,
                        src_created_at,
                        src_updated_at,
                        cnt,
                    ) in zip(
                        source_tags,
                        deprecated_list,
                        format_ids,
                        type_ids,
                        deprecated_flags,
                        created_ats,
                        updated_ats,
                        counts,
                        strict=False,
                    ):
                        canonical_tag = normalize_tag(str(raw_source_tag))
                        canonical_tag_id = tags_mapping.get(canonical_tag)
                        if canonical_tag_id is None:
                            continue

                        fmt_i = int(fmt)
                        tid_i = int(tid) if tid is not None else -1
                        canonical_deprecated = 1 if dep_flag else 0
                        canonical_source_created_at = (
                            str(src_created_at)
                            if src_created_at is not None and str(src_created_at).strip()
                            else None
                        )
                        canonical_source_updated_at = (
                            str(src_updated_at)
                            if src_updated_at is not None and str(src_updated_at).strip()
                            else None
                        )

                        for rec in process_deprecated_tags(
                            canonical_tag=canonical_tag,
                            deprecated_tags=str(dep) if dep is not None else "",
                            format_id=fmt_i,
                            tags_mapping=tags_mapping,
                        ):
                            is_alias = 1 if rec["alias"] else 0
                            st_rows_site.append(
                                (
                                    rec["tag_id"],
                                    rec["format_id"],
                                    tid_i,
                                    rec["alias"],
                                    rec["preferred_tag_id"],
                                    1 if is_alias else canonical_deprecated,
                                    None,  # deprecated_at（不明ならNULL）
                                    canonical_source_created_at if not is_alias else None,
                                )
                            )

                        # usage (canonical のみ)
                        if cnt is not None:
                            try:
                                count_i = int(cnt)
                            except (TypeError, ValueError):
                                continue
                            usage_rows_site.append(
                                (canonical_tag_id, fmt_i, count_i, canonical_source_updated_at)
                            )

                    if st_rows_site:
                        conn.executemany(_TAG_STATUS_UPSERT_IF_CHANGED_SQL, st_rows_site)
                        conn.commit()

                    if usage_rows_site:
                        for tag_id, fmt_i, count_i, ts in usage_rows_site:
                            conn.execute(
                                _TAG_USAGE_COUNTS_UPSERT_IF_GREATER_SQL,
                                (tag_id, fmt_i, count_i, ts, ts),
                            )
                        conn.commit()

                    # translations（存在する言語を全て）
                    present_lang_cols = [c for c in site_chunk.translation_columns if c in df.columns]
                    if present_lang_cols:
                        df_trans = df.select(["source_tag", *present_lang_cols])
                        trans_rows = _extract_translations(df_trans, tags_mapping)
                        if trans_rows:
                            for tag_id, language, translation in trans_rows:
                                conn.execute(
                                    "INSERT OR IGNORE INTO TAG_TRANSLATIONS (tag_id, language, translation) VALUES (?, ?, ?)",
                                    (tag_id, language, translation),
                                )
                            conn.commit()
                            translation_rows_total += len(trans_rows)

                logger.info(
                    f"Imported site_tags: {source_name} (rows={rows_read_total}, translations={translation_rows_total})"
                )
                source_effects.append(
                    {
                        "source": source_name,
                        "action": "imported",
                        "rows_read": int(rows_read_total),
                        "db_changes": int(conn.total_changes - changes_before),
                        "note": "site_tags",
                    }
                )

            # ja/zh/ko の翻訳は、それぞれの必須文字種を含まない場合（絵文字/顔文字/記号/全角記号など）は誤りとして削除する。
            #
            # NOTE:
            # - romaji の作品名など（ASCIIのみ）はデータに含まれない前提（ユーザー判断）
            # - 絵文字は翻訳として不要なため削除対象（ユーザー判断）
            # - 全角コロン等の混入も翻訳としては不要なため削除対象
            total_deleted = 0
            for lang in ("ja", "zh", "ko"):
                changes_before = conn.total_changes
                deleted = _delete_translations_missing_required_script(conn, language=lang)
                if deleted <= 0:
                    continue
                total_deleted += deleted
                source_effects.append(
                    {
                        "source": "TAG_TRANSLATIONS",
                        "action": "cleanup_deleted",
                        "rows_read": 0,
                        "db_changes": int(conn.total_changes - changes_before),
                        "note": f"cleanup_missing_required_script:{lang}",
                    }
                )
                logger.warning(
                    f"[Cleanup] Deleted {deleted} translations lacking required script for {lang}"
                )
            if total_deleted:
                logger.warning(
                    f"[Cleanup] Total deleted translations (required script filter): {total_deleted}"
                )

            changes_before = conn.total_changes
            split_deleted = _split_comma_delimited_translations(conn)
            if split_deleted > 0:
                source_effects.append(
                    {
                        "source": "TAG_TRANSLATIONS",
                        "action": "cleanup_split",
                        "rows_read": 0,
                        "db_changes": int(conn.total_changes - changes_before),
                        "note": "split_comma_delimited_translations",
                    }
                )
                logger.warning(
                    f"[Cleanup] Split {split_deleted} comma-delimited translations in TAG_TRANSLATIONS"
                )

        # Phase 2.5: type/format mapping の不足救済（外部キー整合性）
        _repair_missing_type_format_mapping(conn, report_dir=report_dir_path if report_dir_path else None)

        # Phase 3: インデックス作成
        logger.info("[Phase 3] Creating indexes")
        conn.close()
        build_indexes(output_path)

        # Phase 4: VACUUM/ANALYZE/配布PRAGMA
        logger.info("[Phase 4] Optimizing database (VACUUM/ANALYZE)")
        optimize_database(output_path)

        # 再接続してバージョン情報書き込み
        conn = sqlite3.connect(output_path)
        apply_connection_pragmas(conn, profile="build")
        placeholder_repairs = _repair_placeholder_tag_ids(conn)
        conn.execute(
            "INSERT OR REPLACE INTO DATABASE_METADATA (key, value) VALUES ('version', ?)",
            (version,),
        )
        conn.commit()

        logger.info(f"[COMPLETE] Dataset built successfully: {output_path}")

        # スキップレポート出力（前回実行分が残ると紛らわしいので、常に上書きする）
        if report_dir_path:
            skipped_report = report_dir_path / "skipped_sources.tsv"
            with open(skipped_report, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["source", "reason"])
                writer.writerows(skipped)
            logger.info(f"Skipped sources report: {skipped_report} ({len(skipped)} sources)")
            _write_source_effects_tsv(report_dir_path / "source_effects.tsv", source_effects)

        if report_dir_path:
            _write_placeholder_repairs_tsv(report_dir_path, placeholder_repairs)

    finally:
        conn.close()

    # Parquet出力（任意）: HF Dataset Viewer向けのビュー表
    if parquet_output_dir is not None:
        parquet_dir = Path(parquet_output_dir)
        _export_danbooru_view_parquet(output_path, parquet_dir)


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="Build unified tag database")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output database file path",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("."),
        help="Sources root directory (LoRAIro repo root)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version string",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report output directory (default: <sources>/reports/dataset_builder)",
    )
    parser.add_argument(
        "--include-sources",
        type=Path,
        default=None,
        help="Optional include list file (1 entry per line; supports glob patterns)",
    )
    parser.add_argument(
        "--exclude-sources",
        type=Path,
        default=None,
        help="Optional exclude list file (1 entry per line; supports glob patterns)",
    )
    parser.add_argument(
        "--hf-ja-translation",
        action="append",
        default=None,
        help=(
            "Hugging Face dataset repo_id for JA translations (repeatable). "
            "Example: p1atdev/danbooru-ja-tag-pair-20241015"
        ),
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Optional Parquet output directory (HF Dataset Viewer-friendly view tables).",
    )
    parser.add_argument(
        "--base-db",
        type=Path,
        default=None,
        help=(
            "Base database to copy and build upon (skips Phase 0/1). "
            "Used for MIT version build that extends CC0 version."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output database if it already exists",
    )

    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        sources_dir=args.sources,
        version=args.version,
        report_dir=args.report_dir,
        include_sources_path=args.include_sources,
        exclude_sources_path=args.exclude_sources,
        hf_ja_translation_datasets=args.hf_ja_translation,
        parquet_output_dir=args.parquet_dir,
        base_db_path=args.base_db,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
