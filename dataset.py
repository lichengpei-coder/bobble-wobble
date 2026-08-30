# Creating the dataset
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
data = load_dataset("saberzl/SID_Set")

class AIImagesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes
