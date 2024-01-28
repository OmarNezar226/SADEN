#https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/ultralytics_yolov5.ipynb

import cv2
import torch
from PIL import Image
import pandas as pd

# Model
model = model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=2)

im1 = Image.open('///home/shehab/drone/opengenus_image133.jpeg')  # PIL image

model.conf = 0.25  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
agnostic = False  # NMS class-agnostic
multi_label = False  # NMS multiple labels per box
classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
max_det = 10  # maximum number of detections per image
amp = False  # Automatic Mixed Precision (AMP) inference

results = model(im1, size=320)  # custom inference size

# Inference
results = model([im1], size=640)  # batch of images

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # im1 predictions (tensor)
results.pandas().xyxy[0]  # im1 predictions (pandas)