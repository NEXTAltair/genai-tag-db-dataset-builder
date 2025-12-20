"""Phase 8 CI orchestrator: fetch sources, build datasets, verify, and optionally publish."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from builder_ci.fetcher import fetch_github_repo, fetch_hf_dataset, load_sources_config
from builder_ci.manifest import (
    compare_source_revisions,
    create_build_manifest,
    load_build_manifest,
    should_rebuild,
    write_build_manifest,
)
from builder_ci.publisher import publish_dataset
from builder_ci.readme import generate_readme
_module_root = Path(__file__).resolve().parents[1]
src_dir = _module_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

from genai_tag_db_dataset_builder.builder import build_dataset
from genai_tag_db_dataset_builder.tools.report_db_health import run_health_checks
@dataclass(frozen=True)
class TargetConfig:
    name: str
    repo_id: str
    output_dir: Path
    output_db: Path
    parquet_dir: Path
    report_dir: Path
    manifest_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _external_sources_dir(repo_root: Path) -> Path:
    return repo_root / "external_sources"


def _select_sources_for_target(sources: list[dict], target: str) -> list[dict]:
    selected = []
    for src in sources:
        if not src.get("enabled", True):
            continue
        applies_to = src.get("applies_to", [])
        if target in applies_to:
            selected.append(src)
    return selected


def _fetch_sources(
    sources: list[dict], external_dir: Path, force: bool = False
) -> list[dict]:
    results: list[dict] = []
    for src in sources:
        dest = external_dir / src["id"]
        if src["kind"] == "github":
            meta = fetch_github_repo(src, dest, force=force)
        elif src["kind"] == "hf_dataset":
            meta = fetch_hf_dataset(src, dest, force=force)
        else:
            logger.warning(f"Unknown source kind: {src.get('kind')} ({src.get('id')})")
            continue
        meta["paths_include"] = src.get("paths_include", [])
        meta["data_type"] = src.get("data_type")
        results.append(meta)
    return results


def _generate_include_filter(
    sources: list[dict],
    external_dir: Path,
    report_dir: Path,
    extra_paths: Iterable[str] | None = None,
) -> Path:
    lines: list[str] = []
    for src in sources:
        if src.get("hf_config", {}).get("use_datasets_api"):
            continue
        for pattern in src.get("paths_include", []):
            lines.append(f"{external_dir.name}/{src['id']}/{pattern}")

    if extra_paths:
        lines.extend(extra_paths)

    report_dir.mkdir(parents=True, exist_ok=True)
    include_path = report_dir / "include_sources.generated.txt"
    include_path.write_text("\n".join(lines), encoding="utf-8")
    return include_path


def _stage_translation_csvs(
    sources: list[dict],
    external_dir: Path,
    sources_dir: Path,
) -> list[str]:
    staged: list[str] = []
    staging_root = sources_dir / "TagDB_DataSource_CSV" / "A"

    for src in sources:
        if src.get("data_type") != "translation_ja":
            continue
        if src.get("kind") != "github":
            continue
        src_root = external_dir / src["id"]
        matched = False
        for pattern in src.get("paths_include", []):
            for file_path in src_root.glob(pattern):
                if not file_path.is_file():
                    continue
                staging_root.mkdir(parents=True, exist_ok=True)
                dest_path = staging_root / file_path.name
                shutil.copyfile(file_path, dest_path)
                staged.append(dest_path.relative_to(sources_dir).as_posix())
                matched = True
        if not matched:
            logger.warning(
                f"No translation_ja CSV matched for {src.get('id')} under {src_root}"
            )
    return staged


def _hf_translation_datasets(sources: Iterable[dict]) -> list[str]:
    datasets = []
    for src in sources:
        if src.get("data_type") != "translation_ja":
            continue
        if src.get("hf_config", {}).get("use_datasets_api"):
            datasets.append(src["repo_id"])
    return datasets


def _download_base_db(repo_id: str, dest_dir: Path, force: bool = False) -> dict:
    api = HfApi()
    try:
        info = api.dataset_info(repo_id, repo_type="dataset")
    except TypeError:
        info = api.dataset_info(repo_id)
    revision = info.sha
    sha_path = dest_dir / ".hf_sha"

    if dest_dir.exists() and not force:
        if sha_path.exists():
            existing = sha_path.read_text(encoding="utf-8").strip()
            if existing == revision:
                sqlite_files = sorted(
                    dest_dir.glob("**/*.sqlite"), key=lambda p: p.stat().st_mtime
                )
                if sqlite_files:
                    logger.info(f"Base DB already up to date: {repo_id} ({revision[:8]})")
                    return {
                        "path": sqlite_files[-1],
                        "repo_id": repo_id,
                        "revision": revision,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "skipped": True,
                    }
        logger.info(f"Base DB revision changed or missing, re-downloading: {repo_id}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest_dir,
            allow_patterns=["*.sqlite"],
            revision=revision,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest_dir,
            allow_patterns=["*.sqlite"],
            revision=revision,
        )
    sha_path.write_text(revision, encoding="utf-8")

    sqlite_files = sorted(dest_dir.glob("**/*.sqlite"), key=lambda p: p.stat().st_mtime)
    if not sqlite_files:
        raise FileNotFoundError(f"No sqlite file found in base repo: {repo_id}")

    return {
        "path": sqlite_files[-1],
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "skipped": False,
    }


def _load_health_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    summary: dict[str, str] = {}
    with open(summary_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header != ["metric", "value"]:
            logger.warning(f"Unexpected summary header: {header}")
        for row in reader:
            if len(row) != 2:
                continue
            summary[row[0]] = row[1]
    return summary


def _coerce_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _health_to_manifest(summary: dict) -> tuple[dict | None, dict | None]:
    if not summary:
        return None, None
    stats = {
        "total_tags": _coerce_int(summary.get("total_tags")),
        "total_tag_status": _coerce_int(summary.get("total_tag_status")),
        "total_translations": _coerce_int(summary.get("total_translations")),
        "total_usage_counts": _coerce_int(summary.get("total_usage_counts")),
    }
    health = {
        "quick_check": summary.get("quick_check"),
        "foreign_key_violations": _coerce_int(summary.get("foreign_key_violations")),
        "missing_type_format_mapping_pairs": _coerce_int(
            summary.get("missing_type_format_mapping_pairs")
        ),
        "orphan_tag_status": _coerce_int(summary.get("orphan_tag_status")),
        "orphan_usage_counts": _coerce_int(summary.get("orphan_usage_counts")),
        "orphan_translations": _coerce_int(summary.get("orphan_translations")),
        "duplicate_tags": _coerce_int(summary.get("duplicate_tags")),
        "duplicate_tag_status": _coerce_int(summary.get("duplicate_tag_status")),
        "alias_inconsistencies": _coerce_int(summary.get("alias_inconsistencies")),
        "bad_usage_counts": _coerce_int(summary.get("bad_usage_counts")),
    }
    return stats, health


def _build_target(
    target: TargetConfig,
    sources: list[dict],
    sources_dir: Path,
    external_sources_dir: Path,
    base_db_path: Path,
    base_db_info: dict,
    sources_yml: Path,
    version: str,
    force: bool,
    skip_danbooru_snapshot_replace: bool,
    publish: bool,
    publish_repo_id: str | None,
) -> Path:
    logger.info(f"=== Build start: {target.name} ===")

    source_meta = _fetch_sources(sources, external_sources_dir, force=force)
    staged_paths = _stage_translation_csvs(sources, external_sources_dir, sources_dir)
    include_path = _generate_include_filter(
        sources,
        external_sources_dir,
        target.report_dir,
        extra_paths=staged_paths,
    )
    hf_ja_datasets = _hf_translation_datasets(sources)

    manifest_existing = load_build_manifest(target.manifest_path)
    if not manifest_existing:
        comparison = {"changed": [{"id": "manifest_missing"}], "added": [], "removed": [], "unchanged": []}
    else:
        comparison = compare_source_revisions(manifest_existing, source_meta)
        if (
            base_db_info.get("revision")
            and manifest_existing.get("build_info", {})
            .get("base_db", {})
            .get("revision")
            != base_db_info.get("revision")
        ):
            comparison["changed"].append({"id": "base_db"})
    if not should_rebuild(comparison, force=force):
        logger.info(f"No source changes for {target.name}, skipping build.")
        return target.output_db

    build_dataset(
        output_path=target.output_db,
        sources_dir=sources_dir,
        version=version,
        report_dir=target.report_dir,
        include_sources_path=include_path,
        hf_ja_translation_datasets=hf_ja_datasets,
        parquet_output_dir=target.parquet_dir,
        base_db_path=base_db_path,
        skip_danbooru_snapshot_replace=skip_danbooru_snapshot_replace,
        warn_missing_csv_dir=False,
        overwrite=True,
    )

    summary_path = run_health_checks(target.output_db, target.report_dir / "db_health")
    summary = _load_health_summary(summary_path)
    stats, health = _health_to_manifest(summary)

    manifest = create_build_manifest(
        version=version,
        target=target.name,
        base_db_info=base_db_info,
        sources_metadata=source_meta,
        statistics=stats,
        health_checks=health,
    )
    write_build_manifest(manifest, target.manifest_path)

    readme_path = target.output_dir / "README.md"
    readme_path.write_text(
        generate_readme(
            target=target.name,
            output_dir=target.output_dir,
            sources_yml=sources_yml,
            base_db_info=base_db_info,
        ),
        encoding="utf-8",
    )

    if publish:
        if not publish_repo_id:
            raise ValueError("publish requested but repo_id is not set")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("publish requested but HF_TOKEN is not set")
        publish_dataset(
            output_db=target.output_db,
            output_dir=target.output_dir,
            parquet_dir=target.parquet_dir,
            report_dir=target.report_dir,
            manifest_path=target.manifest_path,
            repo_id=publish_repo_id,
            token=token,
            commit_message=f"{target.name} build {version}",
        )

    logger.info(f"=== Build done: {target.name} ===")
    return target.output_db


def orchestrate(
    target: str,
    sources_yml: Path,
    sources_dir: Path,
    output_root: Path,
    external_sources_dir: Path,
    base_db_dir: Path,
    base_repo_cc0: str,
    version: str,
    force: bool,
    publish: bool,
    repo_cc0: str | None,
    repo_mit: str | None,
    repo_cc4: str | None,
) -> None:
    sources_dir = Path(sources_dir)
    output_root = Path(output_root)
    external_sources_dir = Path(external_sources_dir)
    base_db_dir = Path(base_db_dir)

    sources = load_sources_config(sources_yml)

    def target_cfg(name: str) -> TargetConfig:
        out_dir = output_root / f"out_db_{name}"
        report_dir = out_dir / "report"
        return TargetConfig(
            name=name,
            repo_id=base_repo_cc0,
            output_dir=out_dir,
            output_db=out_dir / f"genai-image-tag-db-{name}.sqlite",
            parquet_dir=out_dir / "parquet_danbooru",
            report_dir=report_dir,
            manifest_path=out_dir / "build_manifest.json",
        )

    base_info = _download_base_db(base_repo_cc0, base_db_dir, force=force)
    base_db_path = Path(base_info["path"])
    base_db_info = {
        "repo_id": base_repo_cc0,
        "revision": base_info.get("revision"),
        "downloaded_at": base_info.get("downloaded_at"),
    }

    def run_cc0() -> Path:
        cc0_sources = _select_sources_for_target(sources, "cc0")
        return _build_target(
            target=target_cfg("cc0"),
            sources=cc0_sources,
            sources_dir=sources_dir,
            external_sources_dir=external_sources_dir,
            base_db_path=base_db_path,
            base_db_info=base_db_info,
            sources_yml=sources_yml,
            version=version,
            force=force,
            skip_danbooru_snapshot_replace=False,
            publish=publish and target in ["cc0", "all"],
            publish_repo_id=repo_cc0,
        )

    def run_mit(cc0_db: Path) -> None:
        mit_sources = _select_sources_for_target(sources, "mit")
        _build_target(
            target=target_cfg("mit"),
            sources=mit_sources,
            sources_dir=sources_dir,
            external_sources_dir=external_sources_dir,
            base_db_path=cc0_db,
            base_db_info={"repo_id": "cc0_local", "path": str(cc0_db)},
            sources_yml=sources_yml,
            version=version,
            force=force,
            skip_danbooru_snapshot_replace=True,
            publish=publish and target in ["mit", "all"],
            publish_repo_id=repo_mit,
        )

    def run_cc4(cc0_db: Path) -> None:
        cc4_sources = _select_sources_for_target(sources, "cc4")
        _build_target(
            target=target_cfg("cc4"),
            sources=cc4_sources,
            sources_dir=sources_dir,
            external_sources_dir=external_sources_dir,
            base_db_path=cc0_db,
            base_db_info={"repo_id": "cc0_local", "path": str(cc0_db)},
            sources_yml=sources_yml,
            version=version,
            force=force,
            skip_danbooru_snapshot_replace=True,
            publish=publish and target in ["cc4", "all"],
            publish_repo_id=repo_cc4,
        )

    if target == "cc0":
        run_cc0()
    elif target == "mit":
        cc0_db = run_cc0()
        run_mit(cc0_db)
    elif target == "cc4":
        cc0_db = run_cc0()
        run_cc4(cc0_db)
    elif target == "all":
        cc0_db = run_cc0()
        run_mit(cc0_db)
        run_cc4(cc0_db)
    else:
        raise ValueError(f"Unknown target: {target}")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8 CI orchestrator")
    p.add_argument(
        "--target",
        choices=["cc0", "mit", "cc4", "all"],
        default="cc0",
        help="Build target",
    )
    p.add_argument(
        "--sources-yml",
        type=Path,
        default=Path(__file__).parent / "sources.yml",
        help="sources.yml path",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=_repo_root(),
        help="base working directory (defaults to repo root)",
    )
    p.add_argument(
        "--sources-dir",
        type=Path,
        default=None,
        help="sources_dir passed to build_dataset (default: --work-dir)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="output root directory (default: --work-dir/ci_output)",
    )
    p.add_argument(
        "--external-sources-dir",
        type=Path,
        default=None,
        help="external sources directory (default: --work-dir/external_sources)",
    )
    p.add_argument(
        "--base-db-dir",
        type=Path,
        default=None,
        help="local directory for cached base DB downloads (default: --work-dir/base_dbs/cc0)",
    )
    p.add_argument(
        "--base-cc0-repo",
        default="NEXTAltair/genai-image-tag-db",
        help="HF dataset repo id for CC0 base DB",
    )
    p.add_argument(
        "--version",
        default=datetime.now(timezone.utc).strftime("v%Y.%m.%d"),
        help="Dataset version string",
    )
    p.add_argument("--force", action="store_true", help="Force rebuild and refetch")
    p.add_argument("--publish", action="store_true", help="Publish to HuggingFace")
    p.add_argument("--repo-cc0", default=None, help="HF repo id for CC0 publish")
    p.add_argument("--repo-mit", default=None, help="HF repo id for MIT publish")
    p.add_argument("--repo-cc4", default=None, help="HF repo id for CC4 publish")

    args = p.parse_args()

    work_dir = args.work_dir
    sources_dir = args.sources_dir or work_dir
    output_root = args.output_root or (work_dir / "ci_output")
    external_sources_dir = args.external_sources_dir or (work_dir / "external_sources")
    base_db_dir = args.base_db_dir or (work_dir / "base_dbs" / "cc0")

    orchestrate(
        target=args.target,
        sources_yml=args.sources_yml,
        sources_dir=sources_dir,
        output_root=output_root,
        external_sources_dir=external_sources_dir,
        base_db_dir=base_db_dir,
        base_repo_cc0=args.base_cc0_repo,
        version=args.version,
        force=args.force,
        publish=args.publish,
        repo_cc0=args.repo_cc0,
        repo_mit=args.repo_mit,
        repo_cc4=args.repo_cc4,
    )


if __name__ == "__main__":
    main()
