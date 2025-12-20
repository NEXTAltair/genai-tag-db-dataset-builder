"""CI README generator."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


LICENSE_BY_TARGET = {
    "cc0": "cc0-1.0",
    "mit": "mit",
    "cc4": "cc-by-4.0",
}


def _load_sources_config(sources_yml: Path) -> list[dict]:
    with open(sources_yml, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    sources = config.get("sources", [])
    return [s for s in sources if s.get("enabled", True)]


def _read_source_effects(report_dir: Path) -> list[dict]:
    effects_path = report_dir / "source_effects.tsv"
    if not effects_path.exists():
        return []
    with open(effects_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames and "source" in reader.fieldnames:
            return list(reader)
        f.seek(0)
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "source": parts[0],
                    "action": parts[1],
                    "rows_read": parts[2],
                    "db_changes": parts[3],
                    "note": parts[4],
                }
            )
        return rows


def _match_source(source_name: str, sources: list[dict]) -> dict | None:
    if source_name.startswith("external_sources/"):
        parts = Path(source_name).parts
        if len(parts) > 1:
            source_id = parts[1]
            for src in sources:
                if src.get("id") == source_id:
                    return src
    if source_name.startswith("hf://datasets/"):
        repo_id = source_name.replace("hf://datasets/", "")
        for src in sources:
            if src.get("repo_id") == repo_id:
                return src
    return None


def generate_readme(
    target: str,
    output_dir: Path,
    sources_yml: Path,
    base_db_info: dict,
) -> str:
    sources = _load_sources_config(sources_yml)
    report_dir = output_dir / "report"
    effects = _read_source_effects(report_dir)
    affected: list[tuple[dict, dict | None]] = []
    for row in effects:
        try:
            changes = int(row.get("db_changes", 0))
        except ValueError:
            changes = 0
        if changes <= 0:
            continue
        src = _match_source(row.get("source", ""), sources)
        affected.append((row, src))

    license_id = LICENSE_BY_TARGET.get(target, "unknown")
    title = f"GenAI Image Tag DB ({license_id})"
    readme = f"""---
license: {license_id}
language:
- en
- ja
tags:
- tag-database
- image-generation
- stable-diffusion
- lora
- booru
- danbooru
- e621
- derpibooru
size_categories:
- 1M<n<10M
task_categories:
- text-retrieval
- text-classification
configs:
- config_name: default
  data_files:
  - split: parquet_danbooru
    path: "parquet_danbooru/*.parquet"
---

# {title}

This repository contains the **{license_id} build** of the tag database.

The main artifact is the **SQLite database**. The `parquet_danbooru/` directory is a derived export so the Hugging Face Dataset Viewer can preview a subset of rows (Danbooru-only).

## Files

- `genai-image-tag-db-{target}.sqlite`: SQLite database
- `parquet_danbooru/*.parquet`: Parquet export for Dataset Viewer
- `build_manifest.json`: Build manifest (revisions and stats)
- `report/`: Source effects and health checks

## Base DB

- Base DB: {base_db_info.get("repo_id", "local")}
- Base info: {base_db_info.get("revision", base_db_info.get("path", "unknown"))}

## Sources affecting this build

Only sources with `db_changes > 0` are listed.
"""

    if not affected:
        readme += "\n- (none)\n"
    else:
        for row, src in affected:
            source_name = row.get("source", "")
            if src:
                url = src.get("url") or src.get("repo_id", "")
                license_name = src.get("license", "unknown")
                readme += f"\n- {src.get('id')} ({license_name}) {url} ({source_name})"
            else:
                readme += f"\n- {source_name}"

    readme += "\n\n## Notes\n\n- This dataset is intended for tag lookup, alias resolution, and translation workflows.\n"
    return readme
