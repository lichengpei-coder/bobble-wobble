# Purpose: Defines the deep learning classifier under the 2B parameter limit.

# Key Responsibilities:
# 1. Loads a vision backbone (such as ConvNeXt-Base or EfficientNet-B4) from timm.
# 2. Replaces the final output head with a single logit output (for binary real vs. fake classification).
