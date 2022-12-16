import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import cv2 as cv

#Read .csv file
landmarks = pd.read_csv('landmark.csv')

path = '51--Dresses/51_Dresses_wearingdress_51_119.jpg'

image = cv.imread(f'WFLW_images/{path}')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

img_ind=0
while img_ind < 7499:
    if landmarks.iloc[img_ind, 0] == path:
        i = 1
        size = 70
        for x in range(0, size):
            y01 = i
            y02 = i + 1
            x  = landmarks.iloc[img_ind,y01]
            y = landmarks.iloc[img_ind,y02]
            image = cv.circle(image, (int(x),int(y)), radius=1, color=(0, 0, 255), thickness=-1)
            i = i + 2
 
        x1 = landmarks.iloc[img_ind,1]
        x2 = landmarks.iloc[img_ind,3]
        y1 = landmarks.iloc[img_ind,2]
        y2 = landmarks.iloc[img_ind,4]

        image = cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=1)
        img_ind = img_ind + 1
    else:
        img_ind = img_ind + 1


cv.imshow('test', image)
cv.waitKey(0)
cv.destroyAllWindows()
