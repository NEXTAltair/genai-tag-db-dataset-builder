-- Add columns required for deepghs/site_tags integration semantics (Phase 6.6).
--
-- Notes:
-- - SQLite does not support `ADD COLUMN IF NOT EXISTS`, so this file is intended
--   for one-time application, or via a wrapper that checks column existence.
-- - `deprecated_at` / `source_created_at` may be NULL when source timestamps are unavailable.

ALTER TABLE TAG_STATUS ADD COLUMN deprecated BOOLEAN NOT NULL DEFAULT 0;
ALTER TABLE TAG_STATUS ADD COLUMN deprecated_at DATETIME NULL;
ALTER TABLE TAG_STATUS ADD COLUMN source_created_at DATETIME NULL;

