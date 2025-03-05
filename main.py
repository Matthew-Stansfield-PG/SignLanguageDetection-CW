from ultralytics import YOLO
import torch
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from logger import Logger

def train():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")
    data_path = "C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/data.yaml"

    add_wandb_callback(model, enable_model_checkpointing=True)
    results = model.train(data=data_path, epochs=3, device=0)
    print(results)
    print("--------")

    model.val()
    model(["C:/Users/Unliv/PycharmProjects/SignLanguageDetection-CW/data/dataset/images/test/C19_jpg.rf.0add6365e398b8d188f428bac97eb868.jpg"])



def main():
    wandb.login(key="4dbce09ff68bb778f03c435cfda8289ee37a28e2")
    wandb.init(project="INM705-CW", name="Sign_language_recognition_model", job_type="training")
    #wandb_logger = Logger(f"Sign_language_recognition_model", project='INM705-CW')#name of model, project name
    #logger = wandb_logger.get_logger()
    train()
    wandb.finish()

if __name__ == '__main__':
    main()