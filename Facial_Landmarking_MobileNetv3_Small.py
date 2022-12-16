import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2 as cv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.utils
import time

'''
The Transform() class resizes all the images to the same size and adjusts the 
landmarks to coincide with the resizing of the images
Input: A csv file containing the data, an image, index for the image and the 
landmarks for said image
Output: The image resized and the landmarks adjusted accordingly
'''
class Transform():
    def __init__(self):
        pass
    '''
    The Crop() function performs the crop on the image using the first two 
    landmarks denoted for the rectangle 
    
    Input: csv file, image, index for image, landmarks
    Output: Resized image, top and left points to adjust the landmarks
    '''
    def Crop(self, csv_file, image, index, landmarks):
        x1 = csv_file.iloc[index, 1]
        y1 = csv_file.iloc[index, 2]
        x2 = csv_file.iloc[index, 3]
        y2 = csv_file.iloc[index, 4]
        xSub = 0
        ySub = 0
        '''
        This for loop and crop function checks to see if there are any landmarks 
        in the dataset that are outside the given rectangle and adjusts the crop 
        accordingly
        '''
        for i in range(0, 136):
            if i%2 == 0:
                if landmarks[i] < x1:
                    if xSub < x1 - landmarks[i]:
                        xSub = x1 - landmarks[i]
            else:
                if landmarks[i] < y1:
                    if ySub < y1 - landmarks[i]:
                        ySub = y1 - landmarks[i]     
        top = y1 - ySub
        left = x1 - xSub
        image = TF.crop(image, int(top), int(left), y2-y1, x2-x1)
        
        return image, top, left
    '''
    The resize() function resizes the images to the preferred size
    
    Input: image, wanted image size
    Output: Newly sized image, size of the new image used to adjust landmarks
    '''
    def resize(self, image, img_size):
        image = TF.resize(image, img_size)
        newSize = image.size
        return image, newSize
    '''
    The ChangeLandmarks() function adjusts the landmarks according to the
    RESIZE 
    
    Input: csv file, initial size of the image, new size of the image, landmarks
    Output: Adjusted landmarks to fit resized images
    '''
    def ChangeLandmarks(self, csv_file, initialSize, newSize, landmarks):
        Xratio = newSize[0]/initialSize[0]
        Yratio = newSize[1]/initialSize[1]
        '''
        This for loop goes through all the landmarks, and changing the landmark
        according to it's positional type (x or y)'
        '''
        for i in range(0, 136):
            if i%2 == 0:
                landmarks[i] = Xratio * landmarks[i]
            else:
                landmarks[i] = Yratio * landmarks[i]
        return landmarks
    '''
    The CropLandmarks() function adjusts the landmarks according to the CROP
    
    Input: top and left points from the Crop() function, landmarks
    Output: Adjusted landmarks to fit the cropped image
    '''
    def CropLandmarks(self, top, left, landmarks):
        for i in range(0, 136):
            if i%2 == 0:
                landmarks[i] = landmarks[i]-left
            else:
                landmarks[i] = landmarks[i]-top
        return landmarks
    '''
    __call__() function
    
    Input: csv_file, image, index of image, landmarks
    Output: Transformed image, image becomes a tensor, adjusted landmarks
    '''
    def __call__(self, csv_file, img, index, landmarks):
        img = Image.fromarray(img)
        img, top, left = self.Crop(csv_file, img, index, landmarks)
        cropSize = img.size
        landmarks = self.CropLandmarks(top, left, landmarks)
        img, newSize = self.resize(img, (224, 224))
        landmarks = self.ChangeLandmarks(csv_file, cropSize, newSize, landmarks)
        img = TF.to_grayscale(img, 1)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5], [0.5])
        return img, landmarks, cropSize

'''
The Face_Dataset() class creates a dataset from the csv file using the
transformation from the Transform() class
Input: csv file, root directory, transformation 
Output: Resized images, adjusted landmarks, both become a tensor
'''
class Face_DataSet():
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv.imread(img_path)
        landmarks = torch.tensor(self.annotations.iloc[index, 5:], dtype=torch.float32)
        x = self.annotations.iloc[index, 1]
        y = self.annotations.iloc[index, 2]
        if self.transform:
            image, landmarks, cropSize = self.transform(self.annotations, image, index, landmarks)
        return image, landmarks, x, y, cropSize

#sets the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Paramaters
num_classes = 136
num_epochs = 20
batch_size = 20
learning_rate = 0.001
error = 2

#Load Data
dataset = Face_DataSet(csv_file = 'landmark.csv', root_dir = 'Dataset', transform = Transform())
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [100000, 7091])
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset= test_dataset, batch_size = batch_size, shuffle = True)
ex = iter(train_loader)
images, landmarks = ex.next()

'''
Used to validate landmarks and images when transformed. No need to use
unless there are changes to the transformation.
Loads 5 sample landmarks from landmarks into tuples and inputs them into imshow 
where image and landmarks are plotted
'''
x = [landmarks[0][0].item(), landmarks[0][2].item(), landmarks[0][4].item(), landmarks[0][6].item(), landmarks[0][8].item(), landmarks[0][10].item()]
y = [landmarks[0][1].item(), landmarks[0][3].item(), landmarks[0][5].item(), landmarks[0][7].item(), landmarks[0][9].item(), landmarks[0][11].item()]

def imshow(img, x, y):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.scatter(x, y)
    plt.show()


#imshow(torchvision.utils.make_grid(images), x, y)


'''
Convulating Neural Netowrk 
5 Convulating Layers
3 Fully Connected Layers
Input: 224x224 Images
Output: 136 landmarks
'''
class Network(nn.Module):
    
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name = 'mobilenetv3'
        self.model = models.mobilenet_v3_small(pretrained = False, num_classes = num_classes)
    def forward(self, x):
        x=self.model(x)
        return x

model = Network().to(device)

#Loss and Optimizer functions
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_min = np.inf

'''
Training loop
'''
for epoch in range(1,num_epochs+1):
    start_time = time.time()
    loss_train = 0
    loss_valid = 0
    running_loss = 0
    
    model.train()
    n_total_steps = len(train_loader)
    for step in range(1,len(train_loader)+1):
    
        images, landmarks = next(iter(train_loader))
        
        images = images.to(device)
        landmarks = landmarks.view(landmarks.size(0),-1).to(device)
        
        predictions = model(images)
        
        # clear all the gradients before calculating them
        optimizer.zero_grad()
        
        # find the loss for the current step
        loss_train_step = criterion(predictions, landmarks)
        
        # calculate the gradients
        loss_train_step.backward()
        
        # update the parameters
        optimizer.step()
        
        loss_train += loss_train_step.item()
        running_loss = loss_train/step       
        
        if step % 100 == 0:
            print(f'Running Training Loss {step}/{n_total_steps}: {running_loss}')
    model.eval() 
    with torch.no_grad():
        n_total_steps = len(test_loader)
        n_samples = batch_size * num_classes
        running_percent = 0
        counter = 0
        for step in range(1,len(test_loader)+1):
            
            images, landmarks, x, y, Size = next(iter(test_loader))
        
            images = images.to(device)
            landmarks = landmarks.view(landmarks.size(0),-1).to(device)
        
            predictions = model(images)

            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step
            
            if step % 100 == 0:
                print(f'Running Validation Loss {step}/{n_total_steps}: {running_loss}')
           
            for j in range(0, batch_size):
                width = Size[0][j]
                height = Size[1][j]
                Xratio = width/224
                Yratio = height/224
                n_correct = 0
                accuracy = 0
                percent = 0
                for i in range(0, 136):
                    if i%2 == 0:
                        landmarks[j][i] = landmarks[j][i] * Xratio
                        landmarks[j][i] = landmarks[j][i] + x[j]
                        predictions[j][i] = predictions[j][i] * Xratio
                        predictions[j][i] = predictions[j][i] + x[j]
                        
                    else:
                        landmarks[j][i] = landmarks[j][i] * Yratio
                        landmarks[j][i] = landmarks[j][i] + y[j]
                        predictions[j][i] = predictions[j][i] * Yratio
                        predictions[j][i] = predictions[j][i] + y[j]
                    validation = predictions[j][i].item() - landmarks[j][i].item()
                    if validation > 0:
                        if validation <= error:
                            n_correct = n_correct + 1
                    if validation < 0:
                        if validation >= -error:
                            n_correct = n_correct + 1
                accuracy = n_correct/n_samples
                percent = 100 * accuracy
                running_percent = running_percent + percent
    average_percent = running_percent/len(test_loader)
    loss_train /= len(train_loader)
    loss_valid /= len(test_loader)
    
    print('\n--------------------------------------------------')
    print(f'Average accuracy: {average_percent}%')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
    print("Total Elapsed Time : {} s".format(time.time()-start_time))
    print('--------------------------------------------------')

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(), './Landmarking_resnet18.pth') 
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')
