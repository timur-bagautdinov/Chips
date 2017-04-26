import cv2
import sys
import os
import numpy as np


def getImList(inputPath, extension):
    return [os.path.join(inputPath, f) for f in os.listdir(inputPath)
            if f.endswith(extension)]


def getHistogramsEqualization(img):
    equ = cv2.equalizeHist(img)
    return equ

def saveImage(img, outputPath, currentImage):
    outputFile = outputPath + '\\' + str(currentImage) + '.bmp'
    print(outputFile)
    cv2.imwrite(outputFile, img)


if __name__ == '__main__':
    inputPath = sys.argv[1]
    outputPath = sys.argv[2]

    imPathList = getImList(inputPath, '.bmp')

    currentImage = 0

    for imPath in imPathList:
        img = cv2.imread(imPath)
        if img is None:
            continue

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.medianBlur(imgGray, 5)
        imgBlurred = cv2.GaussianBlur(imgBlurred, (19, 19), 0, 0)
        #imgBlurred = cv2.GaussianBlur(imgBlurred, (19, 19), 0, 0)
        ret, imgTh = cv2.threshold(imgBlurred, 50, 255, cv2.THRESH_TOZERO)
        imgTh = cv2.adaptiveThreshold(imgTh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY, 11, 0)
        #imgEqu = getHistogramsEqualization(imgTh)

        imgBlurred = cv2.medianBlur(imgTh, 5)
        imgBlurred = cv2.GaussianBlur(imgBlurred, (19, 19), 0, 0)

        imgEdges = cv2.Canny(imgBlurred, 50, 200)

        im2, contours, hierarchy = cv2.findContours(imgEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        big_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 300]

        #cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
        cv2.drawContours(img, big_contours, -1, (0, 255, 255), 2)

        #res = np.hstack((img, cv2.cvtColor(imgEdges, cv2.COLOR_GRAY2BGR)))

        big_diameter = []

        for cnt in big_contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            #radius = int(radius)
            #cv2.circle(img, center, radius, (0, 255, 0), 2)

            if radius * 2 > 150:
                big_diameter.append(cnt)

        cv2.drawContours(img, big_diameter, -1, (0, 0, 255), 2)

        cv2.imshow('Chip', img)

        saveImage(img, outputPath, currentImage)
        currentImage += 1


        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()
