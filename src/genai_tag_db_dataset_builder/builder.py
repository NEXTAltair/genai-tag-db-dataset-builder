"""データセットビルダー（オーケストレーター）.

ローカルにある複数ソース（既存 tags_v4.db / CSV）を統合し、配布用の SQLite DB を生成する。
入力の揺れ（tag 列の意味差・括弧エスケープ・大文字小文字など）は adapter/core 側の正規化に寄せ、
ここでは「取り込み順（再現性）」「衝突の検出とレポート」「DB 作成の一連」を担う。
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

from genai_tag_db_dataset_builder.adapters.csv_adapter import CSV_Adapter
from genai_tag_db_dataset_builder.adapters.tags_v4_adapter import Tags_v4_Adapter
from genai_tag_db_dataset_builder.core.database import (
    apply_connection_pragmas,
    build_indexes,
    create_database,
    optimize_database,
)
from genai_tag_db_dataset_builder.core.exceptions import NormalizedSourceSkipError
from genai_tag_db_dataset_builder.core.master_data import initialize_master_data
from genai_tag_db_dataset_builder.core.merge import merge_tags, normalize_tag, process_deprecated_tags
from genai_tag_db_dataset_builder.core.overrides import ColumnTypeOverrides, load_overrides

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


def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


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
            if translation and isinstance(translation, str) and translation.strip():
                # 言語コードを推定（japanese → ja等）
                language = _infer_language_code(lang_col)
                results.append((tag_id, language, translation.strip()))

    return results


def _infer_language_code(column_name: str) -> str:
    """列名から言語コードを推定する.

    Args:
        column_name: 列名（例: "japanese", "english", "ja", "en"）

    Returns:
        言語コード（例: "ja", "en"）
    """
    col_lower = column_name.lower()
    
    # 直接言語コードの場合
    if col_lower in {"ja", "en", "zh", "ko", "fr", "de", "es", "ru"}:
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


def _find_tags_v4_db(sources_dir: Path) -> Path | None:
    return _first_existing_path(
        [
            sources_dir / "tags_v4.db",
            sources_dir
            / "local_packages"
            / "genai-tag-db-tools"
            / "src"
            / "genai_tag_db_tools"
            / "data"
            / "tags_v4.db",
        ]
    )


def _find_tagdb_csv_root(sources_dir: Path) -> Path | None:
    candidates = [
        sources_dir / "TagDB_DataSource_CSV",
        sources_dir,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            if (c / "A").exists() or any(p.suffix.lower() == ".csv" for p in c.glob("*.csv")):
                return c
    return None


def _iter_csv_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def _infer_format_id(path: Path) -> int:
    name = path.name.lower()
    # e621-rising v2 datasets (tag + deprecated_tags) are derived from e621.
    if "dataset_rising_v2" in name or "rising_v2" in name:
        return 2
    if "danbooru" in name:
        return 1
    if "e621" in name:
        return 2
    if "derpibooru" in name:
        return 3
    return 0


def _infer_repair_mode(path: Path) -> str | None:
    name = path.name.lower()
    if "derpibooru" in name:
        return "derpibooru"
    if "dataset_rising_v2" in name or name == "rising_v2.csv":
        return "dataset_rising_v2"
    if "englishdictionary" in name:
        return "english_dict"
    return None


def _load_source_filters(path: Path | None) -> tuple[set[str], list[str]]:
    """ソースフィルタ（include/exclude）ファイルを読む.

    - 1行1エントリ（相対パス推奨: TagDB_DataSource_CSV/...）
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
    if include_exact or include_patterns:
        if not _match_source(source_name, exact=include_exact, patterns=include_patterns):
            return False
    if exclude_exact or exclude_patterns:
        if _match_source(source_name, exact=exclude_exact, patterns=exclude_patterns):
            return False
    return True


def _read_csv_best_effort(
    path: Path,
    *,
    unknown_report_dir: Path,
    overrides: ColumnTypeOverrides | None = None,
    bad_utf8_report_dir: Path | None = None,
) -> pl.DataFrame | None:
    """CSVを可能な限り読み込む（ヘッダー無しCSVも最低限扱う）."""
    try:
        adapter = CSV_Adapter(
            path,
            repair_mode=_infer_repair_mode(path),
            unknown_report_dir=unknown_report_dir,
            overrides=overrides,
        )
        df = adapter.read()
        if "source_tag" in df.columns:
            return df
    except NormalizedSourceSkipError:
        # 「正則化済みタグ混入ソースは全体スキップ」という運用方針を強制するため、
        # ここではフォールバック読み込みに進まず、呼び出し側へ伝播させる。
        raise
    except Exception as e:
        logger.warning(f"CSV read failed: {path} ({e})")

    # ヘッダー無し CSV の最低限フォールバック:
    # 1列目=tag、2列目=count（数値っぽい場合）として扱う。
    try:
        raw = pl.read_csv(
            path,
            has_header=False,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    except Exception as e:
        msg = str(e).lower()
        if ("utf-8" in msg or "unicode" in msg) and bad_utf8_report_dir is not None:
            raw = _read_csv_skip_invalid_utf8_lines(
                path,
                has_header=False,
                report_dir=bad_utf8_report_dir,
            )
            if raw is None:
                logger.warning(f"CSV fallback read failed: {path} ({e})")
                return None
        else:
            logger.warning(f"CSV fallback read failed: {path} ({e})")
            return None

    if len(raw.columns) < 1:
        return None

    df = raw.rename({raw.columns[0]: "source_tag"})
    if len(raw.columns) >= 2:
        df = df.rename({raw.columns[1]: "count"})
    return df


def _read_csv_skip_invalid_utf8_lines(
    path: Path,
    *,
    has_header: bool,
    report_dir: Path,
) -> pl.DataFrame | None:
    """UTF-8 が「一部行だけ」壊れているCSVを救済する.

    - ファイルをバイト列で行読みし、UTF-8 厳密デコードできない行だけスキップする
    - それ以外の行で Polars 読み込みを再実行する
    - スキップした行はTSVに記録する（後から手動で直せるように）
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "csv_invalid_utf8_lines.tsv"

    kept_lines: list[str] = []
    bad_rows: list[tuple[int, str, str]] = []

    with open(path, "rb") as f:
        for line_no, bline in enumerate(f, start=1):
            try:
                kept_lines.append(bline.decode("utf-8"))
            except UnicodeDecodeError as e:
                preview = bline.decode("utf-8", errors="replace").strip()
                preview = preview.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                bad_rows.append((line_no, str(e), preview))

    if not kept_lines:
        return None

    if bad_rows:
        is_new = not report_path.exists()
        with open(report_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if is_new:
                writer.writerow(["source_file", "line_no", "error", "preview"])
            for line_no, err, preview in bad_rows:
                writer.writerow([str(path), line_no, err, preview])

        logger.warning(f"Skipped {len(bad_rows)} invalid-utf8 lines: {path} -> {report_path}")

    text = "".join(kept_lines)
    try:
        return pl.read_csv(
            io.StringIO(text),
            has_header=has_header,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )
    except Exception as e:
        logger.warning(f"CSV salvage read failed: {path} ({e})")
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
        cur = (type_id, alias, preferred_tag_id)
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = cur
            continue

        if prev != cur:
            conflicts.append(
                _ConflictRow(
                    source=source_name,
                    tag_id=tag_id,
                    format_id=format_id,
                    kind="TAG_STATUS_INPUT_DUPLICATE",
                    existing=f"type_id={prev[0]},alias={prev[1]},preferred_tag_id={prev[2]}",
                    incoming=f"type_id={cur[0]},alias={cur[1]},preferred_tag_id={cur[2]}",
                )
            )

        # 選択ルール: canonical(=alias=0) を優先。次点で preferred_tag_id が小さい方。
        if prev[1] == 0:
            continue
        if cur[1] == 0:
            dedup[key] = cur
            continue
        if cur[2] < prev[2]:
            dedup[key] = cur

    rows = [(k[0], k[1], v[0], v[1], v[2]) for k, v in dedup.items()]

    _ensure_temp_tables(conn)
    conn.execute("DELETE FROM TMP_TAG_STATUS_INPUT;")
    conn.executemany(
        "INSERT INTO TMP_TAG_STATUS_INPUT (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    cur = conn.execute(
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
    ) in cur.fetchall():
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
        for (src, reason) in skipped:
            writer.writerow({"source": src, "reason": reason})


def _repair_placeholder_tag_ids(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """tags_v4 由来の __missing_tag_id_*__ を既存の tag_id に付け替える。

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

    for (missing_id, placeholder_tag) in placeholder_rows:
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
        for (format_id, type_id, alias, preferred_tag_id) in status_rows:
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
        for (format_id, count) in usage_rows:
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
        writer.writerow(["missing_id", "action", "format_id", "preferred_tag_id", "note", "placeholder_tag"])
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

    tags_v4.db に由来するデータで、danbooru/derpibooru/unknown 側に `meta:`/`artist:` が混入することがある。
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

    for (tag_id, format_id, type_id, alias, preferred_tag_id, tag, preferred_tag) in rows:
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
        new_preferred_tag = conn.execute("SELECT tag FROM TAGS WHERE tag_id = ?", (new_preferred,)).fetchone()[0]
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
        logger.warning(
            f"TAG_STATUS conflicts report: {out_path} (0 rows), summary: {out_summary} (0 keys)"
        )
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


def build_dataset(
    output_path: Path | str,
    sources_dir: Path | str,
    version: str,
    report_dir: Path | str | None = None,
    overrides_path: Path | str | None = None,
    include_sources_path: Path | str | None = None,
    exclude_sources_path: Path | str | None = None,
    overwrite: bool = False,
) -> None:
    """データセットをビルドして配布用DBを生成.

    Args:
        output_path: 出力DBファイルのパス
        sources_dir: ソースディレクトリ（tags_v4.db と CSV を含む）
        version: データセットバージョン（例: "v4.1.0"）
        report_dir: レポート出力先ディレクトリ（Noneの場合はレポート出力なし）
        overrides_path: 列タイプオーバーライド設定ファイル（JSON）
        overwrite: 既存のoutput_pathを上書きするか

    Raises:
        FileExistsError: output_pathが既に存在し、overwrite=False の場合
        FileNotFoundError: sources_dirが存在しない場合
    """
    output_path = Path(output_path)
    sources_dir = Path(sources_dir)

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

    # オーバーライド設定読み込み
    overrides = load_overrides(overrides_path) if overrides_path else None

    include_exact, include_patterns = _load_source_filters(Path(include_sources_path)) if include_sources_path else (set(), [])
    exclude_exact, exclude_patterns = _load_source_filters(Path(exclude_sources_path)) if exclude_sources_path else (set(), [])

    # スキップされたソースのレポート（ソース名, 理由）
    skipped: list[tuple[str, str]] = []

    # tags_v4.db 側の不整合（TAG_STATUS が存在しない tag_id / preferred_tag_id を参照）を
    # ビルド側で最低限救済するためのレポート。
    #
    # NOTE:
    # - 欠損 tag_id 自体は「連番の抜け」であり得るが、TAG_STATUS 側が参照している場合は実害がある
    # - ここでは placeholder の TAGS 行を作って外部キー整合性を保ち、あとで手動修正できるように情報を残す
    created_missing_tags: list[dict[str, object]] = []
    placeholder_repairs: list[dict[str, object]] = []

    # Phase 0: DB 作成・マスターデータ登録
    logger.info(f"[Phase 0] Creating database: {output_path}")
    if output_path.exists():
        output_path.unlink()
    create_database(output_path)

    logger.info("[Phase 0] Inserting master data (FORMATS, TYPES, LANGUAGES)")
    initialize_master_data(output_path)

    conn = sqlite3.connect(output_path)
    apply_connection_pragmas(conn, profile="build")
    try:

        # Phase 1: tags_v4.db からベースデータを取り込み
        tags_v4_path = _first_existing_path(
            [
                # 旧: ルート直下
                sources_dir / "tags_v4.db",
                # 旧: CSVディレクトリ配下
                sources_dir / "TagDB_DataSource_CSV" / "tags_v4.db",
                # 現: genai-tag-db-tools の同梱DB（LoRAIro前提）
                sources_dir
                / "local_packages"
                / "genai-tag-db-tools"
                / "src"
                / "genai_tag_db_tools"
                / "data"
                / "tags_v4.db",
            ]
        )
        if tags_v4_path:
            logger.info(f"[Phase 1] Importing tags_v4.db from {tags_v4_path}")
            adapter = Tags_v4_Adapter(tags_v4_path)
            tables = adapter.read()

            df_tags = tables["tags"]
            df_status = tables["tag_status"]
            df_translations = tables["tag_translations"]
            df_usage = tables["tag_usage_counts"]

            if report_dir_path:
                _write_tag_status_conflicts_tsv(report_dir_path, df_status, df_tags)

            # tags_v4.db 側の不整合救済:
            # TAG_STATUS が参照する tag_id / preferred_tag_id が TAGS に存在しない場合がある。
            # 新DB側では外部キー整合性を維持したいので、placeholder TAGS 行を作成して挿入する。
            # ※ ここで作る tag/source_tag は手動修正用のダミー。
            tags_v4_tag_ids = set(df_tags["tag_id"].to_list())
            status_tag_ids = set(df_status["tag_id"].to_list())
            status_preferred_ids = set(df_status["preferred_tag_id"].to_list())
            missing_tag_ids = sorted(status_tag_ids - tags_v4_tag_ids)
            missing_preferred_ids = sorted(status_preferred_ids - tags_v4_tag_ids)
            missing_ids = sorted(set(missing_tag_ids) | set(missing_preferred_ids))

            # tag_id -> tag を tags_v4 の TAGS から引ける範囲で保持（レポート用）
            tags_v4_id_to_tag: dict[int, str] = dict(
                zip(df_tags["tag_id"].to_list(), df_tags["tag"].to_list())
            )

            for missing_id in missing_ids:
                placeholder_tag = f"__missing_tag_id_{missing_id}__"
                conn.execute(
                    "INSERT OR IGNORE INTO TAGS (tag_id, tag, source_tag) VALUES (?, ?, ?)",
                    (missing_id, placeholder_tag, placeholder_tag),
                )

            # 手動修正用の詳細レポート（TAG_STATUSの参照元も含める）
            if missing_ids:
                missing_set = set(missing_ids)
                for row in df_status.to_dicts():
                    tag_id = int(row["tag_id"])
                    preferred_tag_id = int(row["preferred_tag_id"])

                    referenced_as: str | None = None
                    missing_id: int | None = None
                    if tag_id in missing_set:
                        referenced_as = "tag_id"
                        missing_id = tag_id
                    elif preferred_tag_id in missing_set:
                        referenced_as = "preferred_tag_id"
                        missing_id = preferred_tag_id
                    else:
                        continue

                    placeholder_tag = f"__missing_tag_id_{missing_id}__"
                    created_missing_tags.append(
                        {
                            "referenced_as": referenced_as,
                            "missing_id": missing_id,
                            "format_id": int(row["format_id"]),
                            "type_id": int(row["type_id"]),
                            "alias": int(row["alias"]),
                            "tag_id": tag_id,
                            "preferred_tag_id": preferred_tag_id,
                            "tag_if_exists": tags_v4_id_to_tag.get(tag_id, ""),
                            "preferred_tag_if_exists": tags_v4_id_to_tag.get(preferred_tag_id, ""),
                            "placeholder_tag": placeholder_tag,
                        }
                    )

            # TAGS 登録
            existing_tags: set[str] = set()
            for row in df_tags.to_dicts():
                conn.execute(
                    "INSERT INTO TAGS (tag_id, tag, source_tag, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (row["tag_id"], row["tag"], row["source_tag"], row["created_at"], row["updated_at"]),
                )
                existing_tags.add(row["tag"])

            # TAG_STATUS 登録
            for row in df_status.to_dicts():
                conn.execute(
                    "INSERT INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["tag_id"],
                        row["format_id"],
                        row["type_id"],
                        row["alias"],
                        row["preferred_tag_id"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )

            # TAG_TRANSLATIONS 登録
            for row in df_translations.to_dicts():
                conn.execute(
                    "INSERT OR IGNORE INTO TAG_TRANSLATIONS (translation_id, tag_id, language, translation, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row["translation_id"],
                        row["tag_id"],
                        row["language"],
                        row["translation"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )

            # TAG_USAGE_COUNTS 登録
            for row in df_usage.to_dicts():
                conn.execute(
                    "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (row["tag_id"], row["format_id"], row["count"], row["created_at"], row["updated_at"]),
                )

            conn.commit()
            non_e621_prefix_repairs = _repair_non_e621_prefix_preferred(conn)
            if non_e621_prefix_repairs:
                conn.commit()
                if report_dir_path:
                    _write_non_e621_prefix_repairs_tsv(report_dir_path, non_e621_prefix_repairs)
            logger.info(f"[Phase 1] Imported {len(df_tags)} tags from tags_v4.db")

            # 次のtag_id初期値を設定
            cursor = conn.execute("SELECT MAX(tag_id) FROM TAGS")
            max_tag_id = cursor.fetchone()[0]
            next_tag_id = (max_tag_id or 0) + 1
        else:
            logger.warning("[Phase 1] tags_v4.db not found, starting from empty")
            next_tag_id = 1
            existing_tags = set()

        # tag_id -> tag のマッピング（後続のTAG_STATUS/USAGE登録で使用）
        cursor = conn.execute("SELECT tag_id, tag FROM TAGS")
        tags_mapping: dict[str, int] = {row[1]: row[0] for row in cursor.fetchall()}

        # Phase 2: CSV ソースから追加データを取り込み
        csv_dir = sources_dir / "TagDB_DataSource_CSV"
        if not csv_dir.exists():
            logger.warning(f"[Phase 2] CSV directory not found: {csv_dir}, skipping")
        else:
            logger.info(f"[Phase 2] Importing CSV sources from {csv_dir}")

            # CSV ファイルを再帰的に検索（昇順でソート → 再現性確保）
            csv_files = sorted(csv_dir.rglob("*.csv"))
            logger.info(f"[Phase 2] Found {len(csv_files)} CSV files")

            for csv_path in csv_files:
                source_name = csv_path.relative_to(sources_dir).as_posix()

                if not _should_include_source(
                    source_name,
                    include_exact=include_exact,
                    include_patterns=include_patterns,
                    exclude_exact=exclude_exact,
                    exclude_patterns=exclude_patterns,
                ):
                    skipped.append((source_name, "filtered"))
                    continue

                # translation source 検出（language列の存在チェック）
                try:
                    unknown_dir = report_dir_path / "unknown" if report_dir_path else Path("reports/dataset_builder/unknown")
                    df = _read_csv_best_effort(
                        csv_path,
                        unknown_report_dir=unknown_dir,
                        overrides=overrides,
                        bad_utf8_report_dir=report_dir_path,
                    )
                    if df is None:
                        skipped.append((source_name, "read_failed"))
                        continue
                except NormalizedSourceSkipError as e:
                    # NORMALIZED/UNKNOWN判定でスキップ
                    skipped.append((source_name, f"{e.decision}_skipped"))
                    logger.warning(f"Skipped source: {source_name} (decision={e.decision}, signals={e.signals})")
                    continue
                except Exception as e:
                    skipped.append((source_name, f"read_error={e}"))
                    logger.error(f"Failed to read CSV: {csv_path} ({e})")
                    continue

                # 最低限の整合性チェック
                if "source_tag" not in df.columns:
                    skipped.append((source_name, "validation_failed"))
                    continue

                # Translation source 処理（language列または言語別列を持つ場合）
                lang_columns = {col.lower() for col in df.columns} & {"language", "japanese", "english", "chinese", "korean", "ja", "en", "zh", "ko"}
                if lang_columns:
                    trans_rows = _extract_translations(df, tags_mapping)
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
                    if trans_rows:
                        conn.commit()
                        logger.info(
                            f"Imported translations: {source_name} (lang={language}, rows={len(trans_rows)})"
                        )
                    else:
                        logger.warning(f"No translations extracted from {source_name} (tags not yet registered?)")
                    continue

                if "source_tag" not in df.columns and "tag" in df.columns:
                    df = df.rename({"tag": "source_tag"})
                if "source_tag" not in df.columns:
                    skipped.append((source_name, f"missing_source_tag_columns={df.columns}"))
                    continue

                # 補助列の付与（欠損/NULL は既定値で埋める）
                inferred_format_id = _infer_format_id(csv_path)
                if "format_id" not in df.columns:
                    df = df.with_columns(pl.lit(inferred_format_id).cast(pl.Int64).alias("format_id"))
                else:
                    df = df.with_columns(pl.col("format_id").fill_null(inferred_format_id).cast(pl.Int64))

                if "type_id" not in df.columns:
                    df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("type_id"))
                else:
                    df = df.with_columns(pl.col("type_id").fill_null(0).cast(pl.Int64))

                if "deprecated_tags" not in df.columns:
                    df = df.with_columns(pl.lit("").alias("deprecated_tags"))
                else:
                    df = df.with_columns(pl.coalesce([pl.col("deprecated_tags"), pl.lit("")]).alias("deprecated_tags"))

                # chunk 単位で: TAGS 登録 → TAG_STATUS / TAG_USAGE 登録
                chunk_size = 50_000
                for offset in range(0, len(df), chunk_size):
                    chunk = df.slice(offset, chunk_size)

                    # 1) TAGS: source_tag + deprecated_tags を登録（代表 source_tag は先勝ち）
                    candidates = list(_extract_all_tags_from_deprecated(chunk))

                    if candidates:
                        new_tags_df = merge_tags(existing_tags, pl.DataFrame({"source_tag": candidates}), next_tag_id)
                        if len(new_tags_df) > 0:
                            for row in new_tags_df.to_dicts():
                                conn.execute(
                                    "INSERT INTO TAGS (tag_id, tag, source_tag) VALUES (?, ?, ?)",
                                    (row["tag_id"], row["tag"], row["source_tag"]),
                                )
                            conn.commit()

                            added_tags = new_tags_df["tag"].to_list()
                            existing_tags.update(added_tags)
                            tags_mapping.update(dict(zip(new_tags_df["tag"].to_list(), new_tags_df["tag_id"].to_list())))
                            next_tag_id = int(new_tags_df["tag_id"].max()) + 1

                    # 2) TAG_STATUS / TAG_USAGE
                    st_rows: list[tuple[int, int, int, int, int]] = []
                    usage_rows: list[tuple[int, int, int]] = []
                    logged_invalid_count = False

                    source_tags = chunk["source_tag"].to_list()
                    deprecated_list = chunk["deprecated_tags"].to_list()
                    format_ids = chunk["format_id"].to_list()
                    type_ids = chunk["type_id"].to_list()
                    counts = chunk["count"].to_list() if "count" in chunk.columns else [None] * len(chunk)

                    for raw_source_tag, dep, fmt, tid, cnt in zip(
                        source_tags, deprecated_list, format_ids, type_ids, counts
                    ):
                        canonical_tag = normalize_tag(str(raw_source_tag))
                        canonical_tag_id = tags_mapping.get(canonical_tag)
                        if canonical_tag_id is None:
                            # TAGS 登録前提だが、念のため安全側
                            continue

                        fmt_i = int(fmt)
                        tid_i = int(tid) if tid is not None else 0

                        # canonical 自身 + alias 群
                        for rec in process_deprecated_tags(
                            canonical_tag=canonical_tag,
                            deprecated_tags=str(dep) if dep is not None else "",
                            format_id=fmt_i,
                            tags_mapping=tags_mapping,
                        ):
                            st_rows.append(
                                (rec["tag_id"], rec["format_id"], tid_i, rec["alias"], rec["preferred_tag_id"])
                            )

                        # usage (canonical のみ)
                        if cnt is not None:
                            try:
                                count_i = int(cnt)
                            except (TypeError, ValueError):
                                if not logged_invalid_count:
                                    logger.warning(
                                        f"{source_name}: invalid count value encountered: {cnt!r} "
                                        "(skipping count rows for this source)"
                                    )
                                    logged_invalid_count = True
                                continue
                            usage_rows.append((canonical_tag_id, fmt_i, count_i))

                    if st_rows:
                        conn.executemany(
                            "INSERT OR REPLACE INTO TAG_STATUS (tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?, ?, ?, ?, ?)",
                            st_rows,
                        )
                        conn.commit()

                    if usage_rows:
                        for tag_id, format_id, count in usage_rows:
                            conn.execute(
                                "INSERT INTO TAG_USAGE_COUNTS (tag_id, format_id, count) VALUES (?, ?, ?) "
                                "ON CONFLICT(tag_id, format_id) DO UPDATE SET count = MAX(count, excluded.count)",
                                (tag_id, format_id, count),
                            )
                        conn.commit()

                logger.info(f"Imported CSV: {source_name} ({len(df)} rows)")

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
        conn.execute("INSERT INTO DATABASE_METADATA (key, value) VALUES ('version', ?)", (version,))
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

        # tags_v4 の不整合救済レポート出力（手動修正用）
        if created_missing_tags and report_dir_path:
            # NOTE: "created" は誤解を招くため、より説明的なファイル名に変更する。
            # 旧ファイル名も互換のため当面は同内容で出力する。
            out_path = report_dir_path / "tags_v4_missing_tag_references.tsv"
            legacy_out_path = report_dir_path / "missing_tags_created.tsv"
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(
                    [
                        "referenced_as",
                        "missing_id",
                        "format_id",
                        "type_id",
                        "alias",
                        "tag_id",
                        "preferred_tag_id",
                        "tag_if_exists",
                        "preferred_tag_if_exists",
                        "placeholder_tag",
                    ]
                    )
                for r in created_missing_tags:
                    writer.writerow(
                        [
                            r["referenced_as"],
                            r["missing_id"],
                            r["format_id"],
                            r["type_id"],
                            r["alias"],
                            r["tag_id"],
                            r["preferred_tag_id"],
                            r["tag_if_exists"],
                            r["preferred_tag_if_exists"],
                            r["placeholder_tag"],
                        ]
                    )
            if legacy_out_path != out_path:
                legacy_out_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
                logger.warning(
                    f"tags_v4 missing tag references report: {out_path} ({len(created_missing_tags)} rows) "
                    f"(deprecated alias: {legacy_out_path})"
                )
            else:
                logger.warning(
                    f"tags_v4 missing tag references report: {out_path} ({len(created_missing_tags)} rows)"
                )

        if report_dir_path:
            _write_placeholder_repairs_tsv(report_dir_path, placeholder_repairs)

    finally:
        conn.close()


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
        "--overrides",
        type=Path,
        default=None,
        help="Column type overrides JSON (e.g. column_type_overrides.json)",
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
        overrides_path=args.overrides,
        include_sources_path=args.include_sources,
        exclude_sources_path=args.exclude_sources,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
