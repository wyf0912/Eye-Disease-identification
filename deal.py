import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, transform

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


def GetPathfiles(level, path):
    # 所有文件夹，第一个字段是次目录的级别
    dirList = []
    # 所有文件
    fileList = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)
    # 先添加目录级别
    dirList.append(str(level))
    for f in files:
        if (os.path.isdir(path + '/' + f)):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if (f[0] == '.'):
                pass
            else:
                # 添加非隐藏文件夹
                dirList.append(f)
        if (os.path.isfile(path + '/' + f)):
            # 添加文件
            fileList.append(f)
            # 当一个标志使用，文件夹列表第一个级别不打印
    return fileList

def detect_deal(image):
    shape = image.shape
    middle = shape[0] / 2
    middle_y = shape[1] / 2
    if middle>150:
        for j in range(shape[1]):
            if image[int(middle)][j][1] > 10:
                left = j
                break
        for j in range(shape[1] - 1, 0, -1):
            if image[int(middle)][j][1] > 10:
                right = j
                break
        for j in range(shape[0]):
            if image[j][int(middle_y)][1] > 10:
                top = j
                break
        for j in range(shape[0] - 1, 0, -1):
            if image[j][int(middle_y)][1] > 10:
                bottom = j
                break
        print(left, right)
        image = image[top:bottom, left:right]
        image = cv2.resize(image, (200, 200))
        print(top, bottom)
    chans = cv2.split(image)
    chan1 = cv2.equalizeHist(chans[0])
    chan2 = cv2.equalizeHist(chans[1])
    chan3 = cv2.equalizeHist(chans[2])
    image = cv2.merge([chan1, chan2, chan3])
    image = cv2.resize(image,(200,200))
    return image

if __name__ == '__main__':
    path="./datasets/3/"
    resize=1
    filelist = GetPathfiles(1, path)
    for file in filelist:
        print(file[:-4])
        if file[-5:] == '.jepg':
            image = cv2.imread(path+file)
            image = cv2.imread(path+file)
            if resize:
                #print(path+file)
                shape = image.shape
                middle = shape[0] / 2
                middle_y=shape[1] / 2
                for j in range(shape[1]):
                    if image[int(middle)][j][1] > 10:
                        left = j
                        break
                for j in range(shape[1] - 1, 0, -1):
                    if image[int(middle)][j][1] > 10:
                        right = j
                        break
                for j in range(shape[0]):
                    if image[j][int(middle_y)][1] > 10:
                        top= j
                        break
                for j in range(shape[0] - 1, 0, -1):
                    if image[j][int(middle_y)][1] > 10:
                        bottom = j
                        break
                print(left,right)
                image = image[top:bottom, left:right]
                image=cv2.resize(image,(200,200))
                print(top,bottom)
                #print(image)
                #image=cv2.flip(image,0)
            #image=rotate(image, 45)

            chans = cv2.split(image)
            chan1 = cv2.equalizeHist(chans[0])
            chan2 = cv2.equalizeHist(chans[1])
            chan3 = cv2.equalizeHist(chans[2])
            image = cv2.merge([chan1, chan2, chan3])
            cv2.imwrite(path + file, image)
