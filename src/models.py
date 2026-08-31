# Purpose: Defines the deep learning classifier under the 2B parameter limit (spec section 5.3).
#
# Key Responsibilities:
# 1. Loads a vision backbone (e.g. ConvNeXt-Base or EfficientNet-B4) from timm.
# 2. Replaces the final head with a single logit output for binary real (0) vs. fake (1)
#    classification, trained with BCEWithLogitsLoss.
#
# Note: we let timm build the classification head itself (num_classes=1) rather than
# manually slicing off child modules — timm already handles global pooling and the
# correct feature dimensionality per-backbone, which a hand-rolled nn.Sequential slice
# does not (this was the bug in the original prototype's model.py, which hardcoded a
# 1280-d EfficientNet-B7 head that doesn't match B7's actual 2560-d pooled output).

import timm
import torch.nn as nn

# Params measured via timm, well under the 2B cap.
SUPPORTED_BACKBONES = {
    "efficientnet_b4": "~19M params",
    "convnext_tiny": "~28M params",
    "convnext_base": "~88M params",
}


def build_model(backbone: str = "convnext_tiny", pretrained: bool = True) -> nn.Module:
    """
    Builds a binary AIGC-vs-real classifier.

    Args:
        backbone: any timm model name; defaults chosen for a hackathon compute budget.
        pretrained: load ImageNet weights (requires network access on first run).

    Returns:
        nn.Module that outputs a single raw logit per image (shape [batch, 1]).
        Use torch.sigmoid(logits) to get a probability in [0, 1] where values close to
        1 indicate "fake"/AI-generated, matching the predict.py output convention.
    """
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=1)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_under_param_limit(model: nn.Module, limit: int = 2_000_000_000):
    n_params = count_parameters(model)
    if n_params > limit:
        raise ValueError(
            f"Model has {n_params:,} parameters, exceeding the hackathon's {limit:,} limit."
        )
    return n_params
