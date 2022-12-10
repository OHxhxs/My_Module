'''
<얼굴 분류기>
['카리나', '이국주', '마동석']

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


'''

import torch
import torch.nn as nn
import torchvision
from torchvision import models,transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['마동석', '이국주', '카리나']

transforms_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def resnet34_face(image_path):
    model = models.resnet34(weights=True)
    model.fc = nn.Linear(512,3)

    model.load_state_dict(torch.load('C:/Users/HP/Desktop/OHmodule/Classification/model_dict.pth',  map_location=device))      # model.dict.path 잘 설정하기
    model.eval()

    model = model.to(device)
    image = Image.open(image_path)
    # print(image.shape)
    image = transforms_test(image).unsqueeze(0).to(device)    # 이미지 불러와서 무조건! model에 맞게 변환해야함
    # print(image.shape)

    with torch.no_grad():
      outputs = model(image)

    _, preds = torch.max(outputs,1)

    print(preds)
    print(class_names[preds[0]])

    img_predict = class_names[preds[0]]

    return img_predict

if __name__ == "__main__":
    resnet34_face('test.jpg')