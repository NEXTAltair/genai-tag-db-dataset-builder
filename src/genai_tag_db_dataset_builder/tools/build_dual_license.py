from __future__ import annotations

import argparse
from pathlib import Path

from genai_tag_db_dataset_builder.builder import build_dataset


def _read_filter_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _write_union_filters(cc0_filters_path: Path, mit_filters_path: Path, output_path: Path) -> None:
    cc0 = _read_filter_lines(cc0_filters_path)
    mit = _read_filter_lines(mit_filters_path)

    # stable + deterministic
    seen: set[str] = set()
    merged: list[str] = []
    for item in cc0 + mit:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)

    output_path.write_text("\n".join(merged) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CC0 and MIT SQLite/Parquet outputs in one run.")
    parser.add_argument("--sources", type=Path, default=Path("."), help="Sources root directory")
    parser.add_argument("--version", type=str, default="1.0.0", help="Dataset version string")

    parser.add_argument("--cc0-output", type=Path, required=True, help="CC0 SQLite output path")
    parser.add_argument("--mit-output", type=Path, required=True, help="MIT SQLite output path")

    parser.add_argument("--cc0-report-dir", type=Path, required=True, help="CC0 report output directory")
    parser.add_argument("--mit-report-dir", type=Path, required=True, help="MIT report output directory")

    parser.add_argument("--cc0-parquet-dir", type=Path, required=True, help="CC0 parquet output directory")
    parser.add_argument("--mit-parquet-dir", type=Path, required=True, help="MIT parquet output directory")

    parser.add_argument(
        "--include-cc0",
        type=Path,
        required=True,
        help="Include list file for CC0 build (1 entry per line; supports glob patterns)",
    )
    parser.add_argument(
        "--include-mit",
        type=Path,
        required=True,
        help="Include list file for MIT build (MIT-only entries).",
    )
    parser.add_argument(
        "--exclude-sources",
        type=Path,
        default=None,
        help="Optional exclude list file (applies to both builds).",
    )
    parser.add_argument(
        "--hf-ja-translation",
        action="append",
        default=None,
        help="Hugging Face dataset repo_id for JA translations (repeatable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output databases if they already exist",
    )

    args = parser.parse_args()

    args.cc0_report_dir.mkdir(parents=True, exist_ok=True)
    args.mit_report_dir.mkdir(parents=True, exist_ok=True)

    # MIT build = CC0 base + MIT additions (single build run). We generate a merged include file.
    merged_include = args.mit_report_dir / "_include_cc0_plus_mit_sources.generated.txt"
    _write_union_filters(args.include_cc0, args.include_mit, merged_include)

    # CC0 build
    build_dataset(
        output_path=args.cc0_output,
        sources_dir=args.sources,
        version=args.version,
        report_dir=args.cc0_report_dir,
        include_sources_path=args.include_cc0,
        exclude_sources_path=args.exclude_sources,
        hf_ja_translation_datasets=args.hf_ja_translation,
        parquet_output_dir=args.cc0_parquet_dir,
        overwrite=args.overwrite,
    )

    # MIT build (CC0 base + MIT CSV)
    build_dataset(
        output_path=args.mit_output,
        sources_dir=args.sources,
        version=args.version,
        report_dir=args.mit_report_dir,
        include_sources_path=merged_include,
        exclude_sources_path=args.exclude_sources,
        hf_ja_translation_datasets=args.hf_ja_translation,
        parquet_output_dir=args.mit_parquet_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
