import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
from torch import nn
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import time
import multiprocessing

#chooses device for training, cuda better
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')
print("Current device: " +str(device))

#makes results reproducible
manualSeed = 42
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

#if main required to avoid erroring and allow multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()