"""Public loader for detached Chorus package mode."""

from __future__ import annotations

from .inference import ChorusEncoder


def load(
    mode: str = "chorus_3dgs",
    checkpoint: str | None = None,
    outputs=("lang",),
    device: str = "cuda",
    return_numpy: bool = True,
    **options,
) -> ChorusEncoder:
    """Load a Chorus encoder.

    Args:
        mode: Preset name, either `chorus_3dgs` or `chorus_pts`.
        checkpoint: Local checkpoint path or Hugging Face checkpoint filename/stem.
            If omitted, the release checkpoint for `mode` is resolved from the
            configured Hugging Face repo.
        outputs: Feature outputs to return. Supported values are `lang`, `dino`,
            `backbone_upcast`, and `backbone_last`.
        device: CUDA device string. CPU inference is intentionally unsupported.
        return_numpy: Return numpy arrays when true, otherwise CPU torch tensors.
        **options: Optional preset overrides such as `chunk_size`, `output_dir`,
            `outlier_filter`, and `checkpoint_hub`.
    """

    return ChorusEncoder(
        mode=mode,
        checkpoint=checkpoint,
        outputs=outputs,
        device=device,
        return_numpy=return_numpy,
        **options,
    )


__all__ = ["load", "ChorusEncoder"]
