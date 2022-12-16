import numpy as np 
import pandas as pd
import cv2
import os
import random 
import data_process
from torch.utils.data import Dataset 
from augmentation import Rotate_aug, Affine_aug, Mirror, Padding_aug, Img_dropout

symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]

class LandmarkDataset(Dataset):
    def __init__(self, dataset_path, in_res=224):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, header=None).values
        self.inres = (in_res, in_res)
        self.outres = (in_res//8, in_res//8)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        data = self.df[i]
        image_path = data[0]
        # face box
        fb = data[1:5]
        # face landmarks
        lms = data[5:]
        lms = np.reshape(lms, (68, 2))

        img = cv2.imread(image_path)
        cropimg, gtmap = self.process_frame(img, fb, lms)

        return cropimg, gtmap

    
    def process_frame(self, image, fb, lms):
        scale = (fb[2] - fb[0] + fb[3] - fb[1]) / 300
        center = np.array([(fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0])

        cropimg = data_process.crop(image, center, scale, self.inres)
        # transform keypoints
        kp = data_process.transform_kp(lms, center, scale, self.inres)

        # perform augmentation
        if random.uniform(0, 1) > 0.5:
            cropimg, kp = Mirror(cropimg, label=kp, symmetry=symmetry)
        if random.uniform(0, 1) > 0.0:
            angle = random.uniform(-45, 45)
            cropimg, kp = Rotate_aug(cropimg, label=kp, angle=angle)
        if random.uniform(0, 1) > 0.5:
            strength = random.uniform(0, 50)
            cropimg, kp= Affine_aug(cropimg, strength=strength, label=kp)
        if random.uniform(0, 1) > 0.5:
            cropimg = Img_dropout(cropimg, 0.2)
        if random.uniform(0, 1) > 0.5:
            cropimg = Padding_aug(cropimg, 0.3)

        mean = np.array([123.675, 116.28, 103.53], dtype=np.float)
        image = data_process.normalize(cropimg.astype(np.float32), mean)
        image = np.transpose(image, (2, 0, 1))

        kp = kp/8 
        gtmap = np.zeros(shape=(self.outres[0], self.outres[1], 68*3), dtype=float)

        for j in range(68):        
            hm   = self._gaussian_2d(self.outres, kp[j, :], sigma=1)
            gtmap[:, :, j] = hm
            #hm   = np.expand_dims(hm, axis=0)
            offx = self._offset_2dx(self.outres, kp[j, :])
            gtmap[:, :, j+68] = offx
            #offx = np.expand_dims(offx, axis=0)
            offy = self._offset_2dy(self.outres, kp[j, :])
            gtmap[:, :, j+136] = offy
            #offy = np.expand_dims(offy, axis=0)
        gtmap = np.transpose(gtmap, (2, 0, 1))

        return image, gtmap

    
    def _sigmoid(self, p):
        return 1 / (1 + np.exp(-p))

    def _gaussian_2d(self, shape, centre, sigma):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
        alpha = -0.5 / (sigma**2)
        heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
        
        return heatmap

    def _offset_2dx(self, shape, centre):
        shape = tuple(int(sh) for sh in shape) 
        offset_x = np.empty(shape, dtype=np.float32)
        for j in range(shape[0]):
            offset_x[:, j] = self._sigmoid(centre[0] - j) 

        return offset_x

    def _offset_2dy(self, shape, centre):
        shape = tuple(int(sh) for sh in shape) 
        offset_y = np.empty(shape, dtype=np.float32)
        for j in range(shape[1]):
            offset_y[j, :] = self._sigmoid(centre[1] - j) 

        return offset_y

if __name__=="__main__":
    dataset = LandmarkDataset('data/ls3d-landmark.csv')
    print(len(dataset))

    for j in range(len(dataset)):
        image, gtmap = dataset[j]
        print(image.shape, gtmap.shape)

        hm = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                hm[i, j] = np.max(gtmap[:68, i, j]) 

        cv2.imshow('face', image.transpose(1, 2, 0))
        cv2.imshow('hm', cv2.resize(hm, None, fx=8, fy=8))
        cv2.imshow('offx', cv2.resize(gtmap[68, :, :], None, fx=1, fy=1))
        cv2.imshow('offy', cv2.resize(gtmap[2*68, :, :], None, fx=1, fy=1))
        cv2.waitKey(0)
