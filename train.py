# Training the model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from dataset import AIImagesDataset
from model import AIImageDetector
from  robustness import transform

# Dictionary associating target values with folder names
data_dir = load_dataset("saberzl/SID_Set")
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
# Transforming dataset by the robustness
dataset = AIImagesDataset(data_dir, robustness.transform)
# Dataloader
dataloader = Dataloader(dataset, batch_size=32, shuffle=True)

# Split into train/validate/test
train_folder = load_dataset("saberzl/SID_Set", split="train")
validate_folder = load_dataset("saberzl/SID_Set", split="validation")
test_folder = load_dataset("saberzl/SID_Set", split="test")

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
model = AIImageDetector(num_classes=53)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = AIImagesDataset(train_folder, transform=transform)
val_dataset = AIImagesDataset(validate_folder, transform=transform)
test_dataset = AIImagesDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
