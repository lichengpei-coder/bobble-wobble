import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print("System Version:", sys.version)
print("Pytorch Version", torch.__version__)
print("Torchvision Version", torchvision.__version__)
print("Numpy Version", np.__version__)
print("Pandas Version", pd.__version__)
