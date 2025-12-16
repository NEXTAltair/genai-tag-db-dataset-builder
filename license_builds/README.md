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

MITソースのみを入れるため、Phase 1（tags_v4.db）をスキップします。

例:

```powershell
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.builder `
  --output .\out_db_mit\genai_tag_db.sqlite `
  --sources . `
  --report-dir .\out_db_mit `
  --include-sources .\license_builds\include_mit_sources.txt `
  --skip-tags-v4 `
  --overwrite
```
