import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,models,transforms

import numpy as np
import time
from PIL import Image

# Resnet을 위한 것
transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class_names = ['Badger', 'Dog', 'Donkey', 'Goat', 'Mouse', 'Tiger']
class_dict = {'Badger' : '백정', 'Mouse' : '천민', 'Dog' : '광대', 'Donkey' : '상인', 'Goat' : '양반', 'Tiger':'임금'}


def Ornamental(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # cpu

    model = torch.hub.load('C:/Users/HP/Desktop/OHmodule/yolov5', 'custom', path='C:/Users/HP/Desktop/OHmodule/Classification/ornamental_model/yolov5_face_detect_best.pt', source='local')
    result = model(image_path)

    # bounding_box 확인 방법
    # print(result.pandas().xyxy[0])
    img = result.crop(save=True, save_dir='./crop_img')

    model = torch.load('C:/Users/HP/Desktop/OHmodule/Classification/ornamental_model/resnet_ornamental_model.pth', map_location=torch.device('cpu'))
    model.eval()

    model = model.to(device)
    image = Image.open(f'C:/Users/HP/Desktop/OHmodule/crop_img/{image_path}')
    image = transforms_test(image).unsqueeze(0).to(device)    # 이미지 불러와서 무조건! model에 맞게 변환해야함
    # print(image.shape)

    with torch.no_grad():
      outputs = model(image)

      _, preds = torch.max(outputs,1)

      # print(preds)
      print(class_dict[class_names[preds[0]]])

      return class_dict[class_names[preds[0]]]

