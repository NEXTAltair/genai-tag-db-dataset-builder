# License-separated builds (local)

このビルダーは、入力ソースをライセンス単位で分けてDBを生成する前提のための補助ファイルを同梱しています。

## CC0 build

`tags_v4.db` は自家製で `CC0-1.0` とし、Phase 1（tags_v4.db）を含めて構築します。

例:

```powershell
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.builder `
  --output .\out_db_cc0\genai_tag_db.sqlite `
  --sources . `
  --report-dir .\out_db_cc0 `
  --include-sources .\license_builds\include_cc0_sources.txt `
  --overwrite
```

## MIT build

tags_v4.db（CC0基盤）を含め、CC0のCSV + MITのCSVを両方取り込みます。
これにより CC0版とMIT版で tag_id の整合性が維持されます。

例:

```powershell
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.builder `
  --output .\out_db_mit\genai_tag_db.sqlite `
  --sources . `
  --report-dir .\out_db_mit `
  --include-sources .\license_builds\include_mit_sources.txt `
  --overwrite
```

## Dual License Build (Recommended)

CC0版とMIT版を1回の実行で生成するには、`build_dual_license` ツールを使用します：

```powershell
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.tools.build_dual_license `
  --sources . `
  --version 1.0.0 `
  --cc0-output .\out_db_cc0\genai-image-tag-db-cc0.sqlite `
  --mit-output .\out_db_mit\genai-image-tag-db-mit.sqlite `
  --cc0-report-dir .\out_db_cc0 `
  --mit-report-dir .\out_db_mit `
  --cc0-parquet-dir .\out_db_cc0\parquet `
  --mit-parquet-dir .\out_db_mit\parquet `
  --include-cc0 .\license_builds\include_cc0_sources.txt `
  --include-mit .\license_builds\include_mit_sources.txt `
  --overwrite
```
