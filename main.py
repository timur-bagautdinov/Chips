import cv2
import sys
import os
import numpy as np


def getImList(path, extension):
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(extension)]


def getHistogramsEqualization(img):
    equ = cv2.equalizeHist(img)
    return equ


def getThresholding(img):
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 0)
    return th


if __name__ == '__main__':
    inputPath = sys.argv[1]

    imPathList = getImList(inputPath, '.bmp')

    for imPath in imPathList:
        img = cv2.imread(imPath)
        if img is None:
            continue

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #imgBlurred = getFilteredImage(imgGray)
        imgBlurred = cv2.medianBlur(imgGray, 5)
        imgTh = getThresholding(imgBlurred)
        #imgEqu = getHistogramsEqualization(imgTh)

        imgBlurred = cv2.medianBlur(imgTh, 5)
        imgBlurred = cv2.GaussianBlur(imgBlurred, (19, 19), 0, 0)

        imgEdges = cv2.Canny(imgBlurred, 100, 200)

        totalImg = img
        totalImg[imgEdges[:, :] == 255, 0] = 255
        totalImg[imgEdges[:, :] == 255, 1] = 0
        totalImg[imgEdges[:, :] == 255, 2] = 0

        cv2.imshow('Chip', totalImg)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break;

    cv2.destroyAllWindows()
