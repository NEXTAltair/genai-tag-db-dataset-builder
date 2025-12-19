# License-separated builds (local)

このディレクトリは「ライセンス別ビルド」を再現するための include リストを置く場所。

## CC0 build

- CC0版は `tags_v4.db` をベースに、CC0ソースだけを取り込む。
- include: `include_cc0_sources.txt`

```powershell
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.builder `
  --output .\out_db_cc0\genai-image-tag-db-cc0.sqlite `
  --sources . `
  --report-dir .\out_db_cc0 `
  --include-sources .\license_builds\include_cc0_sources.txt `
  --parquet-dir .\out_db_cc0\parquet_danbooru `
  --overwrite
```

## MIT build（CC0版を基にした差分追記）

MIT版は、CC0版SQLiteをベースにして、MITソースのみを追加取り込みする。
これにより、MIT版の `source_effects.tsv` にはMIT差分のみが記録され、READMEのライセンス表記を「実際に影響したMITソース」に絞ることができる。

**前提**: CC0版SQLiteが既にビルド済みであること（上記CC0 buildを先に実行）

```powershell
# MIT版ビルド（CC0版をベースに差分追記）
.\.venv\Scripts\python.exe -m genai_tag_db_dataset_builder.builder `
  --output .\out_db_mit\genai-image-tag-db-mit.sqlite `
  --sources . `
  --report-dir .\out_db_mit `
  --include-sources .\license_builds\include_mit_sources.txt `
  --base-db .\out_db_cc0\genai-image-tag-db-cc0.sqlite `
  --skip-danbooru-snapshot-replace `
  --parquet-dir .\out_db_mit\parquet `
  --overwrite
```

**オプション説明**:
- `--base-db`: CC0版SQLiteをベースとして使用（Phase 0/1をスキップ）
- `--skip-danbooru-snapshot-replace`: Danbooruスナップショット置換をスキップ（CC0版で既に適用済み）
- `--include-sources`: MITソースのみを取り込む

## レポート（READMEのライセンス列挙用）

各ビルドの `report-dir` には `source_effects.tsv` が出力される。

- `db_changes > 0` のソースだけが「実際にDBへ影響したソース」なので、MIT版READMEのライセンス表記対象にできる

