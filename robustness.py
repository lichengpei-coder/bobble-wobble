# Transforming images against the provided augmentations
from torchvision.transforms import v2

transform = v2.Compose((
    v2.JPEG(quality=90),
    v2.GaussianBlur(sigma=0.5),
    v2.Resize((112, 112)),  # target pixel size, not a scale factor
    v2.Resize((224, 224)),
    v2.GuassianNoise(sigma=0.02),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.CenterCrop(179),
))
