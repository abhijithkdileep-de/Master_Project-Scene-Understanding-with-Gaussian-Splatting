from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import torch
from huggingface_hub import hf_hub_download


def resolve_checkpoint_reference(
    checkpoint_ref: str,
    hub_cfg: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
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
        if logger:
            logger.info("Using local checkpoint: %s", resolved["local_path"])
        return resolved

    if hub_cfg is None:
        raise FileNotFoundError(
            f"Checkpoint not found locally and no Hugging Face repo configured: {checkpoint_ref}"
        )
    repo_id = hub_cfg.get("repo_id")
    if not repo_id:
        raise ValueError("checkpoint_hub.repo_id must be set")
    revision = hub_cfg.get("revision", "main")
    filename = checkpoint_ref
    if "/" not in filename and not filename.endswith((".pth", ".pt", ".ckpt")):
        filename = f"{filename}.pth"

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
    if logger:
        logger.info("Resolved Hugging Face checkpoint %s from %s@%s", filename, repo_id, revision)
    return resolved


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, *, strict: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {}
    skipped = {"shape_mismatch": [], "not_in_model": []}
    for key, value in state_dict.items():
        processed_key = key[7:] if key.startswith("module.") else key
        if processed_key in model_state:
            if model_state[processed_key].shape == value.shape:
                filtered[processed_key] = value
            else:
                skipped["shape_mismatch"].append(processed_key)
        else:
            skipped["not_in_model"].append(processed_key)
    info = model.load_state_dict(filtered, strict=strict)
    return dict(load_state_info=info, skipped=skipped)
