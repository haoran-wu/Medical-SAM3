#!/usr/bin/env python3
"""Image backbone helpers for VisiumHD Exp1 patch encoders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


class FlattenBackbone(nn.Module):
    """Wrap a feature extractor and always return [B, C] features."""

    def __init__(self, model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.ndim > 2:
            y = y.flatten(1)
        return y


class CONCHImageBackbone(nn.Module):
    """Adapter that exposes CONCH image features as a plain [B, C] tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(x, proj_contrast=False, normalize=False)


class VirchowImageBackbone(nn.Module):
    """Adapter for Paige Virchow/Virchow2 token outputs."""

    def __init__(self, model: nn.Module, register_tokens: int) -> None:
        super().__init__()
        self.model = model
        self.register_tokens = register_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        class_token = y[:, 0]
        patch_tokens = y[:, 1 + self.register_tokens :]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


class MUSKImageBackbone(nn.Module):
    """Adapter for MUSK image embeddings."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vision_cls, _ = self.model(image=x, return_global=True, with_head=True, out_norm=False)
        return vision_cls


def create_image_backbone(name: str) -> tuple[nn.Module, int]:
    """Create a frozen-feature image backbone by name."""
    key = name.lower().replace("-", "_")

    if key == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(model.children())[:-1])
        return FlattenBackbone(backbone, 2048), 2048

    if key == "gigapath":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "GigaPath requires timm. Install with: pip install timm huggingface_hub"
            ) from exc

        try:
            model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True,
                num_classes=0,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load gated GigaPath weights. Request access to "
                "prov-gigapath/prov-gigapath on Hugging Face and authenticate "
                "in this environment before using --image-backbone gigapath."
            ) from exc
        feature_dim = int(getattr(model, "num_features", 0) or 1536)
        return FlattenBackbone(model, feature_dim), feature_dim

    if key == "uni":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "UNI requires timm. Install with: pip install timm huggingface_hub"
            ) from exc

        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
                num_classes=0,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load gated UNI weights. Request access to MahmoodLab/UNI "
                "on Hugging Face with an institutional email account and authenticate "
                "in this environment before using --image-backbone uni."
            ) from exc
        feature_dim = int(getattr(model, "num_features", 0) or 1024)
        return FlattenBackbone(model, feature_dim), feature_dim

    if key in {"virchow", "virchow2"}:
        try:
            import timm
            from timm.layers import SwiGLUPacked
        except ImportError as exc:
            raise ImportError(
                "Virchow requires timm>=0.9.11 with SwiGLUPacked and huggingface_hub."
            ) from exc

        repo = "paige-ai/Virchow2" if key == "virchow2" else "paige-ai/Virchow"
        register_tokens = 4 if key == "virchow2" else 0
        try:
            model = timm.create_model(
                f"hf-hub:{repo}",
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load gated {repo} weights. Request access on Hugging Face "
                "and authenticate in this environment before using this backbone."
            ) from exc
        return VirchowImageBackbone(model, register_tokens=register_tokens), 2560

    if key == "musk":
        try:
            from timm.models import create_model
        except ImportError as exc:
            raise ImportError(
                "MUSK requires timm. Install with: pip install timm huggingface_hub"
            ) from exc

        third_party_musk = Path(__file__).resolve().parents[2] / "third_party" / "MUSK"
        if third_party_musk.exists() and str(third_party_musk) not in sys.path:
            sys.path.insert(0, str(third_party_musk))

        try:
            from musk import modeling, utils  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MUSK requires the official repo at third_party/MUSK plus its lightweight "
                "dependencies. Clone https://github.com/lilab-stanford/MUSK into "
                "third_party/MUSK and install fairscale, sentencepiece, and ftfy."
            ) from exc

        try:
            model = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        except Exception as exc:
            raise RuntimeError(
                "Could not load gated MUSK weights from xiangjx/musk. Request/accept "
                "access on Hugging Face and authenticate this HPC environment before "
                "using --image-backbone musk."
            ) from exc

        return MUSKImageBackbone(model), 1024

    if key == "conch":
        try:
            from huggingface_hub import HfFolder
        except ImportError as exc:
            raise ImportError(
                "CONCH requires huggingface_hub. Install with: pip install huggingface_hub"
            ) from exc

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as exc:
            raise ImportError(
                "CONCH requires the official package. Install with: "
                "pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from exc

        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or HfFolder.get_token()
        )
        try:
            model, _ = create_model_from_pretrained(
                "conch_ViT-B-16",
                "hf_hub:MahmoodLab/CONCH",
                hf_auth_token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load gated CONCH weights. Request access to MahmoodLab/CONCH "
                "on Hugging Face with an institutional email account, install the "
                "official CONCH package, and authenticate in this environment before "
                "using --image-backbone conch."
            ) from exc

        backbone = CONCHImageBackbone(model)
        with torch.inference_mode():
            feature_dim = int(backbone(torch.zeros(1, 3, 224, 224)).shape[-1])
        return backbone, feature_dim

    raise ValueError(f"Unsupported image backbone: {name}")
