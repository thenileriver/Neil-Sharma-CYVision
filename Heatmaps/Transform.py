import torchvision.transforms.functional as TF
from PIL import Image

'''
The Crop() function performs the crop on the image using the first two 
landmarks denoted for the rectangle 

Input: csv file, image, index for image, landmarks
Output: Resized image, top and left points to adjust the landmarks
'''
def Crop(csv_file, image, index, landmarks):
    x1 = csv_file[0]
    y1 = csv_file[1]
    x2 = csv_file[2]
    y2 = csv_file[3]
    xSub = 0
    ySub = 0
    '''
    This for loop and crop function checks to see if there are any landmarks 
    in the dataset that are outside the given rectangle and adjusts the crop 
    accordingly
    '''
    for i in range(0, 68):
        if landmarks[i][0] < x1:
            if xSub < x1 - landmarks[i][0]:
                xSub = x1 - landmarks[i][0]
        if landmarks[i][1] < y1:
            if ySub < y1 - landmarks[i][1]:
                ySub = y1 - landmarks[i][1]     
    top = y1 - ySub
    left = x1 - xSub

    img = Image.fromarray(image)
    img = TF.crop(img, int(top), int(left), y2-y1, x2-x1)
    
    return img, top, left
'''
The resize() function resizes the images to the preferred size

Input: image, wanted image size
Output: Newly sized image, size of the new image used to adjust landmarks
'''
def resize(image, img_size):
    image = TF.resize(image, img_size)
    newSize = image.size
    return image, newSize
'''
The ChangeLandmarks() function adjusts the landmarks according to the
RESIZE 

Input: csv file, initial size of the image, new size of the image, landmarks
Output: Adjusted landmarks to fit resized images
'''
def ChangeLandmarks(csv_file, initialSize, newSize, landmarks):
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
def CropLandmarks(top, left, landmarks):
    for i in range(0, 136):
        if i%2 == 0:
            landmarks[i] = landmarks[i]-left
        else:
            landmarks[i] = landmarks[i]-top
    return landmarks
