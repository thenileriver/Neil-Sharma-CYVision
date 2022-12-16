import os
import numpy as np
import cv2
import random
import math


def Rotate_aug(src, angle, label=None, center=None, scale=1.0):
    image = src
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if label is None:
        for i in range(image.shape[2]):
            image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=[127.,127.,127.])
        return image, None
    else:
        label = label.T
        full_M = np.row_stack((M, np.asarray([0, 0, 1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=[127.,127.,127.])
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(full_M, full_label)
        label_rotated = label_rotated[0:2, :]
        label_rotated = label_rotated.T
        return img_rotated, label_rotated


def Perspective_aug(src, strength, label=None):
    image = src
    pts_base = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts1 = np.random.rand(4, 2) * random.uniform(-strength, strength) + pts_base
    pts1 = pts1.astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts_base)
    trans_img = cv2.warpPerspective(image, M, (src.shape[1], src.shape[0]))
    label_rotated = None
    if label is not None:
        label = label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        label_rotated = label_rotated.astype(np.int32)
        label_rotated = label_rotated.T
    return trans_img, label_rotated


def Affine_aug(src, strength, label=None):
    image = src
    pts_base = np.float32([[10, 100], [200, 50], [100, 250]])
    pts1 = np.random.rand(3, 2) * random.uniform(-strength, strength) + pts_base
    pts1 = pts1.astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts_base)
    trans_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                               borderValue=[127.,127.,127.])
    label_rotated = None
    if label is not None:
        label = label.T
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(M, full_label)
        label_rotated = label_rotated.T
    return trans_img, label_rotated


def Padding_aug(src, max_pattern_ratio=0.05):
    src = src.astype(np.float32)
    pattern = np.ones_like(src)
    ratio = random.uniform(0, max_pattern_ratio)
    height, width, _ = src.shape
    if random.uniform(0, 1) > 0.5:
        if random.uniform(0, 1) > 0.5:
            pattern[0:int(ratio * height), :, :] = 0
        else:
            pattern[height - int(ratio * height):, :, :] = 0
    else:
        if random.uniform(0, 1) > 0.5:
            pattern[:, 0:int(ratio * width), :] = 0
        else:
            pattern[:, width - int(ratio * width):, :] = 0
    bias_pattern = (1 - pattern) * [127.,127.,127.]
    img = src * pattern + bias_pattern
    img = img.astype(np.uint8)
    return img


def Blur_aug(src, ksize=(3, 3)):
    for i in range(src.shape[2]):
        src[:, :, i] = cv2.GaussianBlur(src[:, :, i], ksize, 1.5)
    return src


def Img_dropout(src, max_pattern_ratio=0.05):
    width_ratio = random.uniform(0, max_pattern_ratio)
    height_ratio = random.uniform(0, max_pattern_ratio)
    width = src.shape[1]
    height = src.shape[0]
    block_width = width * width_ratio
    block_height = height * height_ratio
    width_start = int(random.uniform(0, width - block_width))
    width_end = int(width_start + block_width)
    height_start = int(random.uniform(0, height - block_height))
    height_end = int(height_start + block_height)
    src[height_start:height_end, width_start:width_end, :] = np.array([127.,127.,127.], dtype=src.dtype)
    return src


def Mirror(src, label=None, symmetry=None):
    img = cv2.flip(src, 1)
    if label is None:
        return img, label

    width = img.shape[1]
    cod = []
    allc = []
    for i in range(label.shape[0]):
        x, y = label[i][0], label[i][1]
        if x >= 0:
            x = width - 1 - x
        cod.append((x, y))
    for (q, w) in symmetry:
        cod[q], cod[w] = cod[w], cod[q]
    for i in range(label.shape[0]):
        allc.append(cod[i][0])
        allc.append(cod[i][1])
    label = np.array(allc).reshape(label.shape[0], 2)
    return img, label
