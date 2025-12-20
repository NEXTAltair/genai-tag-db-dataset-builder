"""builder_ci: Phase 8 CI/CD統合レイヤ.

外部ソースの取得、ビルドマニフェスト生成、オーケストレーションを提供する。
"""

from builder_ci.fetcher import (
    fetch_all_sources,
    fetch_github_repo,
    fetch_hf_dataset,
    load_sources_config,
)
from builder_ci.manifest import (
    compare_source_revisions,
    create_build_manifest,
    load_build_manifest,
    should_rebuild,
    write_build_manifest,
)

__version__ = "0.1.0"

__all__ = [
    # fetcher
    "load_sources_config",
    "fetch_github_repo",
    "fetch_hf_dataset",
    "fetch_all_sources",
    # manifest
    "create_build_manifest",
    "write_build_manifest",
    "load_build_manifest",
    "compare_source_revisions",
    "should_rebuild",
]
