import torch
from torch.xpu import device
from ultralytics import YOLO
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


#print(torch.cuda.get_device_name(0))

# model = YOLO("yolov8n.pt")
# data_path = "C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/data.yaml"
#
# results = model.train(data=data_path, epochs=100)
# results = model.val()
# results = model("C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/images/test/C19_jpg.rf.0add6365e398b8d188f428bac97eb868.jpg")
