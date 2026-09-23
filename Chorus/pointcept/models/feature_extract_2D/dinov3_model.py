import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Normalize
from transformers import AutoModel

LOGGER = logging.getLogger(__name__)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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


def image_from_path(
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


class DINOv3_Wrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        pretrained_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        model_source = pretrained_weights or model_name
        self.feature_model = AutoModel.from_pretrained(
            model_source,
            dtype=torch.float32,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.feature_model.eval()
        for param in self.feature_model.parameters():
            param.requires_grad = False

        config = self.feature_model.config
        self.patch_size = config.patch_size
        self.num_register_tokens = getattr(config, "num_register_tokens", 0)
        self.hidden_size = config.hidden_size
        LOGGER.info(
            "Initialized DINOv3 teacher %s with patch_size=%s, register_tokens=%s, hidden_size=%s",
            model_source,
            self.patch_size,
            self.num_register_tokens,
            self.hidden_size,
        )

    def _patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() != 4:
            raise ValueError(f"Expected image batch of shape (B, C, H, W); got {tuple(x.shape)}")
        _, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"DINOv3 expects 3-channel RGB input, received {channels} channels")

        height_aligned = (height // self.patch_size) * self.patch_size
        width_aligned = (width // self.patch_size) * self.patch_size
        if height_aligned <= 0 or width_aligned <= 0:
            raise ValueError(
                f"Image size {(height, width)} is smaller than patch size {self.patch_size}"
            )
        if height_aligned != height or width_aligned != width:
            x = nn.functional.interpolate(
                x,
                size=(height_aligned, width_aligned),
                mode="bilinear",
                align_corners=False,
            )

        with torch.inference_mode():
            outputs = self.feature_model(x)
        tokens = outputs.last_hidden_state[:, 1 + self.num_register_tokens :, :]
        patch_h = height_aligned // self.patch_size
        patch_w = width_aligned // self.patch_size
        expected_tokens = patch_h * patch_w
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DINOv3 token count {tokens.shape[1]} does not match patch grid {patch_h}x{patch_w}"
            )
        return tokens, patch_h, patch_w

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 4,
        return_class_token: bool = True,
        reshape: bool = True,
    ):
        del n, return_class_token
        tokens, patch_h, patch_w = self._patch_tokens(x)
        if reshape:
            tokens = tokens.unflatten(1, (patch_h, patch_w)).permute(0, 3, 1, 2)
        return [tokens]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, patch_h, patch_w = self._patch_tokens(x)
        return tokens.unflatten(1, (patch_h, patch_w)).permute(0, 3, 1, 2)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected single image tensor of shape (C, H, W); got {tuple(x.shape)}")
        return self.forward(x[None])[0]
