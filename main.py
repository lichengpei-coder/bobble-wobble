import torch
import torchvision
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt # Data visualisation
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

from datasets import load_dataset
from dataset import AIImagesDataset
from huggingface_hub import login
import os

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

# HuggingFace Authentication
HF_TOKEN = os.environ.get("HF_TOKEN")
login(token=HF_TOKEN)

# Obtaining dataset
ai_images_dataset = AIImagesDataset(data_dir=load_dataset("saberzl/SID_Set"))
print(len(ai_images_dataset))
image, label = ai_images_dataset[0]
