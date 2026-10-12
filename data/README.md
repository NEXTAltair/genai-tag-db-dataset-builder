# Legacy tag-table enrichment bundle

`tag_table_bundle.json` contains the portable tag translations and search categories extracted from the
prepared legacy tag-table review files. It contains 3,364 canonical tags, 3,685 Japanese translation
strings, 409 category nodes, and 3,796 tag-to-category assignments covering 3,303 tags.

The bundle combines the element tables from `AL_NovelAI-Prompt Generator` and `NovelAI-Prompt Generator
のコピー` with `呪文管理203`'s saved-prompt table. Source references retain the relative TSV path and
one-based spreadsheet row. Exact duplicate references are stored once. The two search taxonomies use
separate namespaces: `prompt-generator` and `prompt-manager-203`.

The portable format has `schema_version: 1` and `bundle_id: "legacy-tag-table"`. Its records are:

- `tags`: canonical `tag`, representative raw `source_tag`, `translations`, and `source_references`.
- `categories`: stable `category_id`, `namespace`, nullable `parent_id`, `label`, `depth`, and the full
  category `path`.
- `assignments`: canonical `tag`, `category_id`, and `source_references`.
- Each source reference contains only `source_path` and the positive integer `source_row`.

Tag strings and assignments use the builder's existing `core.normalize.normalize_tag` function.
Normalization changes 107 of the 3,370 reviewed spellings and merges six pairs, retaining the combined
translations and references. The export preserves existing category IDs and hierarchy rather than
inferring booru tag types from search categories.

The exporter accepts the prepared `primary_candidates.json`, `categories.tsv`, and
`tag_categories.tsv` files. To reproduce the checked-in bundle from a local review directory, run:

```console
python -m genai_tag_db_dataset_builder.tools.export_tag_table_bundle \
  --review-dir /path/to/tag_table_review \
  --output data/tag_table_bundle.json
```

Records and references are sorted, duplicate values are merged, and JSON is UTF-8 with LF line endings
and a trailing newline. The command prints record counts and the output SHA-256. The checked-in bundle
has SHA-256 `4e8a305e9d59fc23e107dec6201f404cb822489e87e4549621b77555caed3d59`.

Flagged candidates, composite prompts, weighting syntax, and LoRA helper presets are excluded. The
prepared classifications already exclude provisional registration, Google Translate, favorites, and
preset categories. Database comparisons, existing translations, preferred-tag guesses, private cache
paths, and other analysis metadata are excluded from the portable format.
