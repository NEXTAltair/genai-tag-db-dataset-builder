"""Metadata generation for dataset.

This module generates comprehensive metadata about the built dataset,
including statistics, schema information, and build details.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def generate_metadata(
    db_path: Path | str,
    output_path: Path | str,
    version: str = "1.0.0",
) -> dict:
    """Generate metadata for the dataset.

    Args:
        db_path: Path to the built database
        output_path: Output path for metadata JSON
        version: Dataset version string

    Returns:
        Metadata dictionary
    """
    db_path = Path(db_path)
    output_path = Path(output_path)

    logger.info(f"Generating metadata for {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        # Basic statistics
        tags_count = conn.execute("SELECT COUNT(*) FROM TAGS").fetchone()[0]
        formats_count = conn.execute("SELECT COUNT(*) FROM TAG_FORMATS").fetchone()[0]
        type_names_count = conn.execute("SELECT COUNT(*) FROM TAG_TYPE_NAME").fetchone()[0]
        translations_count = conn.execute("SELECT COUNT(*) FROM TAG_TRANSLATIONS").fetchone()[0]

        # Per-format statistics
        format_stats = []
        formats = conn.execute("SELECT format_id, format_name FROM TAG_FORMATS ORDER BY format_id").fetchall()
        for format_id, format_name in formats:
            count = conn.execute(
                "SELECT COUNT(DISTINCT tag_id) FROM TAG_STATUS WHERE format_id = ?", (format_id,)
            ).fetchone()[0]
            format_stats.append({"format_id": format_id, "format_name": format_name, "tag_count": count})

        # Database file size
        db_size_bytes = db_path.stat().st_size
        db_size_mb = db_size_bytes / (1024 * 1024)

        # Build metadata
        metadata = {
            "version": version,
            "build_date": datetime.now(UTC).isoformat(),
            "schema_version": "1.0",
            "statistics": {
                "total_tags": tags_count,
                "total_formats": formats_count,
                "total_type_names": type_names_count,
                "total_translations": translations_count,
                "database_size_bytes": db_size_bytes,
                "database_size_mb": round(db_size_mb, 2),
                "format_statistics": format_stats,
            },
            "tables": {
                "TAGS": {
                    "description": "Unified tag definitions",
                    "columns": ["tag_id", "tag", "source_tag", "created_at", "updated_at"],
                },
                "TAG_STATUS": {
                    "description": "Tag metadata per format",
                    "columns": [
                        "tag_id",
                        "format_id",
                        "type_id",
                        "alias",
                        "preferred_tag_id",
                        "created_at",
                        "updated_at",
                    ],
                },
                "TAG_TRANSLATIONS": {
                    "description": "Tag translations",
                    "columns": ["translation_id", "tag_id", "language", "translation", "created_at", "updated_at"],
                },
                "TAG_USAGE_COUNTS": {
                    "description": "Tag usage statistics per format",
                    "columns": ["tag_id", "format_id", "count", "created_at", "updated_at"],
                },
                "TAG_FORMATS": {
                    "description": "Format definitions (master data)",
                    "columns": ["format_id", "format_name", "description"],
                },
                "TAG_TYPE_NAME": {
                    "description": "Type name definitions (master data)",
                    "columns": ["type_name_id", "type_name", "description"],
                },
                "TAG_TYPE_FORMAT_MAPPING": {
                    "description": "Type ID to type name mapping per format (master data)",
                    "columns": ["format_id", "type_id", "type_name_id", "description"],
                },
            },
        }

        # Write metadata
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata written to {output_path}")
        logger.info(f"Total tags: {tags_count:,}")
        logger.info(f"Database size: {db_size_mb:.2f} MB")

        return metadata

    finally:
        conn.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate dataset metadata")
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the built database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for metadata JSON",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version string",
    )

    args = parser.parse_args()

    generate_metadata(
        db_path=args.db,
        output_path=args.output,
        version=args.version,
    )


if __name__ == "__main__":
    main()
