"""Checkpoint resolution helpers for standalone inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from huggingface_hub import hf_hub_download


def resolve_checkpoint_reference(
    checkpoint_ref: str,
    hub_cfg: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    if not checkpoint_ref:
        raise ValueError("checkpoint_ref must be a non-empty path or checkpoint name")

    checkpoint_path = Path(checkpoint_ref).expanduser()
    if checkpoint_path.is_file():
        resolved = dict(
            source="local",
            requested=checkpoint_ref,
            resolved_name=checkpoint_path.name,
            local_path=str(checkpoint_path.resolve()),
        )
        if logger is not None:
            logger.info("Using local checkpoint: %s", resolved["local_path"])
        return resolved

    if hub_cfg is None:
        raise FileNotFoundError(
            f"Checkpoint not found locally and no Hugging Face repo configured: {checkpoint_ref}"
        )

    repo_id = hub_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("checkpoint_hub.repo_id must be set to resolve non-local checkpoints")

    revision = hub_cfg.get("revision", "main")
    filename = checkpoint_ref
    if "/" not in checkpoint_ref and not filename.endswith((".pth", ".pt", ".ckpt")):
        filename = f"{checkpoint_ref}.pth"

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=hub_cfg.get("repo_type", "model"),
    )
    resolved = dict(
        source="huggingface",
        requested=checkpoint_ref,
        resolved_name=filename,
        local_path=local_path,
        repo_id=repo_id,
        revision=revision,
    )
    if logger is not None:
        logger.info(
            "Resolved Hugging Face checkpoint %s from %s@%s",
            filename,
            repo_id,
            revision,
        )
    return resolved
