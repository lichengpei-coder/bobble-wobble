import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm

import matplotlib.pyplot as plt # Data visualisation
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

from datasets import load_dataset

# print("System Version:", sys.version)
# print("Pytorch Version", torch.__version__)
# print("Torchvision Version", torchvision.__version__)
# print("Numpy Version", np.__version__)
# print("Pandas Version", pd.__version__)

from dataset import AIImagesDataset
from huggingface_hub import login
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
login(token=HF_TOKEN)
ai_images_dataset = AIImagesDataset(data_dir=load_dataset("saberzl/SID_Set"))
print(len(ai_images_dataset))
