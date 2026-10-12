"""Add legacy tag translations and independent search taxonomies to a build."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from genai_tag_db_dataset_builder.core.normalize import normalize_tag


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceReference(_Record):
    source_path: str = Field(min_length=1)
    source_row: StrictInt = Field(ge=1)

    @field_validator("source_path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts or "\\" in value:
            raise ValueError("source_path must be a relative POSIX path")
        return value


def _single_tag(value: str) -> str:
    if (
        not value
        or len(value) > 90
        or not value.isascii()
        or any(char in value for char in ",\n\r[]{}<>#")
        or "((" in value
        or "))" in value
        or re.search(r":\s*[\d.]", value)
        or normalize_tag(value) != value
    ):
        raise ValueError("expected a canonical single tag, without prompt syntax")
    return value


class TagRecord(_Record):
    tag: str
    source_tag: str = Field(min_length=1)
    translations: list[str]
    source_references: list[SourceReference]

    _canonical_tag = field_validator("tag")(_single_tag)


class CategoryRecord(_Record):
    category_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    parent_id: str | None
    label: str = Field(min_length=1)
    depth: StrictInt = Field(ge=1)
    path: list[str] = Field(min_length=1)


class AssignmentRecord(_Record):
    tag: str
    category_id: str
    source_references: list[SourceReference]


class EnrichmentBundle(_Record):
    schema_version: StrictInt
    bundle_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    tags: list[TagRecord]
    categories: list[CategoryRecord]
    assignments: list[AssignmentRecord]

    @model_validator(mode="after")
    def validate_links(self) -> EnrichmentBundle:
        if self.schema_version != 1:
            raise ValueError("unsupported enrichment schema_version")
        tags = {row.tag for row in self.tags}
        categories = {row.category_id: row for row in self.categories}
        paths = {(row.namespace, tuple(row.path)) for row in self.categories}
        if (
            len(tags) != len(self.tags)
            or len(categories) != len(self.categories)
            or len(paths) != len(categories)
        ):
            raise ValueError("duplicate tag, category ID or category path")
        for row in self.categories:
            if row.depth != len(row.path) or row.path[-1] != row.label or not all(row.path):
                raise ValueError("category depth/label/path mismatch")
            if row.depth == 1:
                if row.parent_id is not None:
                    raise ValueError("root category cannot have a parent")
            else:
                parent = categories.get(row.parent_id or "")
                if parent is None or parent.namespace != row.namespace or parent.path != row.path[:-1]:
                    raise ValueError("invalid category parent")
        seen: set[tuple[str, str]] = set()
        for assignment in self.assignments:
            key = (assignment.tag, assignment.category_id)
            if assignment.tag not in tags or assignment.category_id not in categories or key in seen:
                raise ValueError("invalid or duplicate category assignment")
            seen.add(key)
        return self


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS TAG_CATEGORIES (
        category_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, parent_id TEXT,
        label TEXT NOT NULL, depth INTEGER NOT NULL CHECK(depth >= 1), path TEXT NOT NULL,
        UNIQUE(namespace, path), FOREIGN KEY(parent_id) REFERENCES TAG_CATEGORIES(category_id))""",
    """CREATE TABLE IF NOT EXISTS TAG_CATEGORY_MEMBERSHIPS (
        tag_id INTEGER NOT NULL, category_id TEXT NOT NULL,
        PRIMARY KEY(tag_id, category_id), FOREIGN KEY(tag_id) REFERENCES TAGS(tag_id),
        FOREIGN KEY(category_id) REFERENCES TAG_CATEGORIES(category_id))""",
    """CREATE TABLE IF NOT EXISTS TAG_SOURCE_REFERENCES (
        bundle_id TEXT NOT NULL, tag_id INTEGER NOT NULL, source_path TEXT NOT NULL,
        source_row INTEGER NOT NULL CHECK(source_row >= 1),
        PRIMARY KEY(bundle_id, tag_id, source_path, source_row), FOREIGN KEY(tag_id) REFERENCES TAGS(tag_id))""",
    """CREATE TABLE IF NOT EXISTS TAG_CATEGORY_SOURCES (
        bundle_id TEXT NOT NULL, tag_id INTEGER NOT NULL, category_id TEXT NOT NULL,
        source_path TEXT NOT NULL, source_row INTEGER NOT NULL CHECK(source_row >= 1),
        PRIMARY KEY(bundle_id, tag_id, category_id, source_path, source_row),
        FOREIGN KEY(tag_id, category_id) REFERENCES TAG_CATEGORY_MEMBERSHIPS(tag_id, category_id))""",
    "CREATE INDEX IF NOT EXISTS idx_category_parent ON TAG_CATEGORIES(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_category_members ON TAG_CATEGORY_MEMBERSHIPS(category_id, tag_id)",
)


@dataclass
class EnrichmentResult:
    bundle_id: str
    sha256: str
    input_tags: int
    added_tags: int = 0
    added_translations: int = 0
    added_categories: int = 0
    added_memberships: int = 0
    db_changes: int = 0
    translation_notes: list[dict[str, object]] = field(default_factory=list)


def load_bundle(path: Path) -> tuple[EnrichmentBundle, str]:
    data = path.read_bytes()
    return EnrichmentBundle.model_validate_json(data), hashlib.sha256(data).hexdigest()


def apply_enrichment(
    conn: sqlite3.Connection, bundle_path: Path, report_dir: Path | None = None
) -> EnrichmentResult:
    """Validate before mutation, then atomically add records; existing Japanese wins.

    No alias inference or translation replacement is performed. A translated alias
    target in any format also prevents filling the alias with an old sheet label.
    """
    bundle, digest = load_bundle(bundle_path)
    result = EnrichmentResult(bundle.bundle_id, digest, len(bundle.tags))
    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "off":
        if conn.in_transaction:
            raise ValueError("enrichment requires a rollback journal; commit before switching OFF mode")
        conn.execute("PRAGMA journal_mode=MEMORY")
    before = conn.total_changes
    conn.execute("SAVEPOINT tag_table_enrichment")
    try:
        for statement in SCHEMA:
            conn.execute(statement)
        conn.execute("INSERT OR IGNORE INTO TAG_FORMATS VALUES (0, 'unknown', '')")
        conn.execute("INSERT OR IGNORE INTO TAG_TYPE_NAME VALUES (0, 'unknown', '')")
        conn.execute("INSERT OR IGNORE INTO TAG_TYPE_FORMAT_MAPPING VALUES (0, 0, 0, '')")
        for row in sorted(bundle.tags, key=lambda item: item.tag):
            hit = conn.execute("SELECT tag_id FROM TAGS WHERE tag=?", (row.tag,)).fetchone()
            if hit is None:
                cursor = conn.execute(
                    "INSERT INTO TAGS(source_tag, tag) VALUES (?, ?)", (row.source_tag, row.tag)
                )
                tag_id = cursor.lastrowid
                conn.execute(
                    "INSERT INTO TAG_STATUS(tag_id, format_id, type_id, alias, preferred_tag_id) VALUES (?,0,0,0,?)",
                    (tag_id, tag_id),
                )
                result.added_tags += 1
            else:
                tag_id = hit[0]
            conn.executemany(
                "INSERT OR IGNORE INTO TAG_SOURCE_REFERENCES VALUES (?, ?, ?, ?)",
                [
                    (bundle.bundle_id, tag_id, ref.source_path, ref.source_row)
                    for ref in row.source_references
                ],
            )
            has_ja = conn.execute(
                "SELECT 1 FROM TAG_TRANSLATIONS WHERE tag_id=? AND language='ja' AND trim(translation) != '' LIMIT 1",
                (tag_id,),
            ).fetchone()
            alias_ja = conn.execute(
                """SELECT 1 FROM TAG_STATUS s JOIN TAG_TRANSLATIONS t ON t.tag_id=s.preferred_tag_id
                WHERE s.tag_id=? AND s.alias=1 AND s.deprecated=0 AND t.language='ja'
                AND trim(t.translation) != '' LIMIT 1""",
                (tag_id,),
            ).fetchone()
            translations = sorted({text.strip() for text in row.translations if text.strip()})
            if has_ja or alias_ja:
                result.translation_notes.append(
                    {"tag": row.tag, "reason": "existing_ja" if has_ja else "alias_target_ja"}
                )
                continue
            for translation in translations:
                if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", translation):
                    result.translation_notes.append(
                        {"tag": row.tag, "reason": "missing_japanese_script", "value": translation}
                    )
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO TAG_TRANSLATIONS(tag_id,language,translation) VALUES (?,'ja',?)",
                    (tag_id, translation),
                )
                result.added_translations += cursor.rowcount
            if len(translations) > 1:
                result.translation_notes.append(
                    {"tag": row.tag, "reason": "multiple_source_values", "values": translations}
                )

        for category in sorted(bundle.categories, key=lambda item: (item.depth, item.category_id)):
            values = (
                category.category_id,
                category.namespace,
                category.parent_id,
                category.label,
                category.depth,
                json.dumps(category.path, ensure_ascii=False),
            )
            existing = conn.execute(
                "SELECT * FROM TAG_CATEGORIES WHERE category_id=?", (category.category_id,)
            ).fetchone()
            if existing is not None and tuple(existing) != values:
                raise ValueError(f"category ID conflicts with existing hierarchy: {category.category_id}")
            cursor = conn.execute("INSERT OR IGNORE INTO TAG_CATEGORIES VALUES (?, ?, ?, ?, ?, ?)", values)
            if existing is None and cursor.rowcount == 0:
                raise ValueError(f"category path conflicts with existing ID: {category.category_id}")
            result.added_categories += cursor.rowcount
        for assignment in bundle.assignments:
            tag_id = conn.execute("SELECT tag_id FROM TAGS WHERE tag=?", (assignment.tag,)).fetchone()[0]
            cursor = conn.execute(
                "INSERT OR IGNORE INTO TAG_CATEGORY_MEMBERSHIPS VALUES (?, ?)",
                (tag_id, assignment.category_id),
            )
            result.added_memberships += cursor.rowcount
            conn.executemany(
                "INSERT OR IGNORE INTO TAG_CATEGORY_SOURCES VALUES (?, ?, ?, ?, ?)",
                [
                    (bundle.bundle_id, tag_id, assignment.category_id, ref.source_path, ref.source_row)
                    for ref in assignment.source_references
                ],
            )
        conn.execute(
            """INSERT INTO DATABASE_METADATA(key,value) VALUES (?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value WHERE value != excluded.value""",
            (f"tag_table_bundle:{bundle.bundle_id}", digest),
        )
        conn.execute("RELEASE SAVEPOINT tag_table_enrichment")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT tag_table_enrichment")
        conn.execute("RELEASE SAVEPOINT tag_table_enrichment")
        raise
    result.db_changes = conn.total_changes - before
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"tag_table_{bundle.bundle_id}.json").write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result
