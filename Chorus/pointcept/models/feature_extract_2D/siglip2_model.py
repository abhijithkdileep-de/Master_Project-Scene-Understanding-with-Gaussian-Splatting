import logging
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel

LOGGER = logging.getLogger(__name__)


def image_from_path_siglip(
    image_path: str,
    resize_down=None,
    normalize: bool = False,
    crop_edge: int = 0,
):
    del resize_down, normalize, crop_edge
    return Image.open(image_path).convert("RGB")


class Siglip2_Wrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip2-large-patch16-512",
        pretrained_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        model_source = pretrained_weights or model_name
        self.feature_model = SiglipVisionModel.from_pretrained(model_source)
        self.processor = AutoProcessor.from_pretrained(model_source)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model.to(self.device)
        self.feature_model.eval()
        for param in self.feature_model.parameters():
            param.requires_grad = False

        config = self.feature_model.config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        LOGGER.info(
            "Initialized SigLIP2 teacher %s with patch_size=%s, hidden_size=%s",
            model_source,
            self.patch_size,
            self.hidden_size,
        )

    def _reshape_tokens(
        self,
        tokens: torch.Tensor,
        pixel_height: int,
        pixel_width: int,
    ) -> torch.Tensor:
        patch_h = pixel_height // self.patch_size
        patch_w = pixel_width // self.patch_size
        expected_tokens = patch_h * patch_w
        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:, :]
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"SigLIP2 token count {tokens.shape[1]} does not match patch grid {patch_h}x{patch_w}"
            )
        return tokens.unflatten(1, (patch_h, patch_w)).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected image batch of shape (B, C, H, W); got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"SigLIP2 expects 3-channel RGB input, received {x.shape[1]} channels")
        x = x.to(self.device, dtype=self.feature_model.dtype)
        height, width = x.shape[2], x.shape[3]
        height_aligned = (height // self.patch_size) * self.patch_size
        width_aligned = (width // self.patch_size) * self.patch_size
        if height_aligned <= 0 or width_aligned <= 0:
            raise ValueError(f"Image size {(height, width)} is smaller than patch size {self.patch_size}")
        if height_aligned != height or width_aligned != width:
            x = nn.functional.interpolate(
                x,
                size=(height_aligned, width_aligned),
                mode="bilinear",
                align_corners=False,
            )
        with torch.inference_mode():
            outputs = self.feature_model(pixel_values=x)
        return self._reshape_tokens(outputs.last_hidden_state, height_aligned, width_aligned)

    def predict(self, image) -> torch.Tensor:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.feature_model(**inputs)
        pixel_values = inputs["pixel_values"]
        return self._reshape_tokens(
            outputs.last_hidden_state,
            pixel_values.shape[2],
            pixel_values.shape[3],
        )[0]
