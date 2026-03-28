from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from .constants import (
    DEFAULT_HF_REVISION_ENV,
    DEFAULT_HF_TOKEN_ENV,
    DEFAULT_MODEL_REPO_ENV,
    LOCAL_CACHE_DIR,
    PROJECT_ROOT,
)


def _is_real_candidate(path: Path) -> bool:
    return path.is_file() and "Zone.Identifier" not in path.name


def resolve_weight_path(
    checkpoint_name: str,
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    hf_model_repo_id: str | None = None,
    hf_revision: str | None = None,
    hf_token: str | None = None,
) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if _is_real_candidate(candidate):
            return candidate
        raise FileNotFoundError(f"Weight file not found: {candidate}")

    for candidate in (
        PROJECT_ROOT / checkpoint_name,
        LOCAL_CACHE_DIR / checkpoint_name,
    ):
        if _is_real_candidate(candidate):
            return candidate

    repo_id = hf_model_repo_id or os.getenv(DEFAULT_MODEL_REPO_ENV)
    if not repo_id:
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' was not found locally and HF model repo is not configured."
        )

    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        repo_type="model",
        revision=hf_revision or os.getenv(DEFAULT_HF_REVISION_ENV),
        token=hf_token or os.getenv(DEFAULT_HF_TOKEN_ENV),
        local_dir=LOCAL_CACHE_DIR,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded).resolve()
