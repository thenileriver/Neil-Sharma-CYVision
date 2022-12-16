import numpy as np
import scipy
import cv2

def transform(point, center, scale, res, invert=0):
    resolution = res[0]
    pt = np.array ([point[0], point[1], 1.0])
    h = 200.0 * scale
    m = np.eye(3)
    m[0,0] = resolution / h
    m[1,1] = resolution / h
    m[0,2] = resolution * (-center[0] / h + 0.5)
    m[1,2] = resolution * (-center[1] / h + 0.5)
    if invert:
        m = np.linalg.inv(m)

    return np.matmul(m, pt)[0:2]

def crop(image, center, scale, res):
    resolution = res[0]
    ul = transform([1, 1], center, scale, res, invert=1).astype(np.int)
    br = transform([resolution, resolution], center, scale, res, invert=1).astype(np.int)

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg

def normalize(imgdata, color_mean):
    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    imgdata = imgdata / np.array([58.395, 57.120, 57.375], dtype=np.float)

    return imgdata

def transform_kp(joints, center, scale, res):
    newjoints = np.copy(joints)
    for i in range(joints.shape[0]):
        if joints[i, 0] > 0 and joints[i, 1] > 0:
            _x = transform(newjoints[i, 0:2] + 1, center=center, scale=scale, res=res, invert=0)
            newjoints[i, 0:2] = _x
    return newjoints

