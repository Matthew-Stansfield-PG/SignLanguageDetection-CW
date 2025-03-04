import torch
from torch.xpu import device
from ultralytics import YOLO

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

print(torch.__version__)
print(torch.version.cuda)
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)
# print(torch.cuda.get_device_name(0))
#
# model = YOLO("yolov8n.pt")
# data_path = "C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/data.yaml"
#
# results = model.train(data=data_path, epochs=1, device=0)