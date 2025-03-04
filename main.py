from ultralytics import YOLO
import torch
from logger import Logger

import torch
from ultralytics import YOLO



def train(logger):
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")
    data_path = "C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/data.yaml"

    results = model.train(data=data_path, epochs=3, device="cpu")

def main():
    wandb_logger = Logger(f"Sign_language_recognition_model", project='INM705-CW')#name of model, project name
    logger = wandb_logger.get_logger()
    train(logger)

if __name__ == '__main__':
    main()