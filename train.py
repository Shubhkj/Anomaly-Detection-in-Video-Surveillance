import wandb

# Log in with your Wandb API key
wandb.login(key='0e1ff96ff8b65d3ed566fdd3df9f7d36a77fc47a')
from ultralytics import YOLO

# loading a pre-trained model
# if the first time loading a model, it will first download the model in the directory
# available pre-trained models are YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
model = YOLO("D:\\test (2)\\yolov8n.pt")

# will throw an exception if false
model._check_is_pytorch_model()

data_yaml_path = "data.yaml"

# Use 'cpu' for device since you don't have CUDA available
model.train(data=data_yaml_path,
            epochs=100,
            imgsz=100,
            device='cpu')


import matplotlib.pyplot as plt
import os
import random
from PIL import Image as PILImage

plt.figure(figsize = (20, 50))
for i in range(5):
    plt.subplot(1,5,i+1)
    sample_path = random.choice(os.listdir("runs/detect/train/weights"))
    img = PILImage.open("./runs/detect/predict5/" + sample_path)
    plt.imshow(img)

plt.tight_layout()


