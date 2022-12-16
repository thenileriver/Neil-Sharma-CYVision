import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
import random

csv_file = pd.read_csv('landmark.csv')
random_index = random.randint(0, len(csv_file))
face_index = random_index
weights_path = 'Landmarking_Mobile_Small.pth'
csv_file = pd.read_csv('landmark.csv')
image_path = f'Dataset/{csv_file.iloc[face_index, 0]}'
#image_path = 'Dataset/LS3D-W/300VW-3D/Trainset/204/0111.jpg'
validation = csv_file.iloc[face_index, 5:]
error = 2
batch_size = 1000

class Resnet(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

class MobileSmall(nn.Module):
    
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name = 'mobilenetv3'
        self.model = models.mobilenet_v3_small(pretrained = False, num_classes = num_classes)
    def forward(self, x):
        x=self.model(x)
        return x


class MobileLarge(nn.Module):
    
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name = 'mobilenetv3'
        self.model = models.mobilenet_v3_large(pretrained = False, num_classes = num_classes)
    def forward(self, x):
        x=self.model(x)
        return x


best_network = MobileSmall()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False) 
best_network.eval()

# HERE
image = cv2.imread(image_path)
#grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


x = csv_file.iloc[face_index, 1]
y = csv_file.iloc[face_index, 2]
w = csv_file.iloc[face_index, 3]
h = csv_file.iloc[face_index, 4]

if x < 0:
    x = 0
if y < 0:
    y = 0

all_landmarks = []
image = display_image[y:h, x:w]      #MobileNet
#image = grayscale_image[y:h, x:w]   #Resnet

height, width, _ = image.shape
Xratio = width/224
Yratio = height/224
image = TF.resize(Image.fromarray(image), size=(224, 224))
image = TF.to_tensor(image)
image = TF.normalize(image, [0.5], [0.5])

with torch.no_grad():
    landmarks = best_network(image.unsqueeze(0)) 

for i in range(0, 136):
    if i%2 == 0:
        landmarks[0][i] = landmarks[0][i] * Xratio
        landmarks[0][i] = landmarks[0][i] + x
    else:
        landmarks[0][i] = landmarks[0][i] * Yratio
        landmarks[0][i] = landmarks[0][i] + y

plt.figure()
plt.imshow(display_image)

index = 0
for i in range (0, 68):
    plt.scatter(landmarks[0][index], landmarks[0][index+1], c = 'c', s = 1)
    index = index + 2

plt.show()
n_correct = 0
n_landmarks = 136
for i in range(0, 136):
    valid = validation[i]
    prediction = landmarks[0][i].item()
    test = valid - prediction
    #print(f'Index {i+1}: prediction->{prediction}  vs  test->{valid}')
    if test < 0:
        if test >= -error:
            n_correct = n_correct + 1
    
    if test > 0:
        if test <= error:
            n_correct = n_correct + 1

accuracy = n_correct/n_landmarks
percent = accuracy * 100
print(f'Accuracy on Image: {percent}%')
