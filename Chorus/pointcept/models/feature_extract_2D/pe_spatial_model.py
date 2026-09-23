import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Normalize

LOGGER = logging.getLogger(__name__)

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def _pil_to_torch(pil_image: Image.Image, resolution: Tuple[int, int]) -> torch.Tensor:
    image = pil_image.resize(resolution)
    image = torch.from_numpy(np.array(image)).float() / 255.0
    if image.ndim == 3:
        return image.permute(2, 0, 1)
    return image.unsqueeze(dim=-1).permute(2, 0, 1)


def _resolve_resolution(
    image_size: Tuple[int, int],
    resize_down: Optional[Union[int, Tuple[int, int], list]],
    crop_edge: int,
) -> Tuple[int, int]:
    width, height = image_size
    longest = max(width, height)
    if isinstance(resize_down, (list, tuple)):
        return int(resize_down[0] + 2 * crop_edge), int(resize_down[1] + 2 * crop_edge)
    if resize_down and longest > resize_down + 2 * crop_edge:
        scale = longest / (resize_down + 2 * crop_edge)
    else:
        scale = 1
    return int(width / scale), int(height / scale)


def image_from_path_pe_spatial(
    image_path: str,
    resize_down: Optional[Union[int, Tuple[int, int], list]] = 1600,
    normalize: bool = False,
    crop_edge: int = 0,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    resolution = _resolve_resolution(image.size, resize_down, crop_edge)
    tensor = _pil_to_torch(image, resolution)
    if normalize:
        tensor = Normalize(MEAN, STD)(tensor)
    if crop_edge > 0:
        tensor = tensor[:, crop_edge:-crop_edge, crop_edge:-crop_edge]
    return tensor.cuda()


def _resolve_perception_root(explicit_root: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    env_root = os.environ.get("PERCEPTION_MODELS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    current = Path(__file__).resolve()
    candidates.append(current.parents[2] / "perception_models")
    candidates.append(Path.home() / "repos" / "perception_models")
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _import_vision_transformer(perception_root: Optional[str] = None):
    module_name = "core.vision_encoder.pe"
    try:
        module = __import__(module_name, fromlist=["VisionTransformer"])
        return module.VisionTransformer
    except ImportError:
        root = _resolve_perception_root(perception_root)
        if root and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        module = __import__(module_name, fromlist=["VisionTransformer"])
        return module.VisionTransformer
    except ImportError as exc:  # pragma: no cover - requires local repo setup
        raise ImportError(
            "Could not import VisionTransformer from perception_models. "
            "Set PERCEPTION_MODELS_ROOT or pass perception_root to PE_Spatial_Wrapper."
        ) from exc


class PE_Spatial_Wrapper(nn.Module):
    """Thin wrapper around the optional PE-Spatial VisionTransformer."""

    def __init__(
        self,
        model_name: str = "PE-Spatial-L14-448",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        perception_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        VisionTransformer = _import_vision_transformer(perception_root)
        cfg_name = model_name.split("/")[-1]
        self.feature_model = VisionTransformer.from_config(
            cfg_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model.to(self.device)
        self.feature_model.eval()
        for param in self.feature_model.parameters():
            param.requires_grad = False

        self.patch_size = getattr(self.feature_model, "patch_size", 14)
        self.hidden_size = getattr(self.feature_model, "width", None)
        if self.hidden_size is None:
            raise AttributeError("VisionTransformer is expected to expose a `width` attribute")
        LOGGER.info(
            "Initialized PE-Spatial teacher %s with patch_size=%s, hidden_size=%s",
            cfg_name,
            self.patch_size,
            self.hidden_size,
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self.predict(x)
        if x.dim() != 4:
            raise ValueError(f"Expected image tensor of shape (C,H,W) or (B,C,H,W); got {tuple(x.shape)}")
        return torch.stack([self.predict(image) for image in x], dim=0)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected single image tensor of shape (C, H, W); got {tuple(x.shape)}")
        channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"PE-Spatial expects 3-channel RGB input, received {channels} channels")

        patch = self.patch_size
        height_aligned = (height // patch) * patch
        width_aligned = (width // patch) * patch
        if height_aligned <= 0 or width_aligned <= 0:
            raise ValueError(f"Image size {(height, width)} is smaller than patch size {patch}")
        if height_aligned != height or width_aligned != width:
            x = nn.functional.interpolate(
                x[None],
                size=(height_aligned, width_aligned),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        tokens = self.feature_model.forward_features(
            x.to(self.device)[None],
            strip_cls_token=True,
        )
        if tokens.dim() != 3 or tokens.shape[0] != 1:
            raise RuntimeError(f"Unexpected PE-Spatial token shape: {tuple(tokens.shape)}")

        patch_h = height_aligned // patch
        patch_w = width_aligned // patch
        if tokens.shape[1] != patch_h * patch_w:
            raise RuntimeError(
                f"PE-Spatial token count {tokens.shape[1]} does not match patch grid {patch_h}x{patch_w}"
            )
        return tokens[0].view(patch_h, patch_w, -1).permute(2, 0, 1).contiguous()
