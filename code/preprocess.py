import cv2
import numpy as np
import matplotlib.pyplot as plt

class Preprocess:
    def __init__(self, file):
        self.file = file
        
    def fitSize(width, height):
        image = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE)
        image_resized = cv2.resize(image, (W,H))
        return image_resized

    def removeNoise():
        image_bgr = cv2.imread(self.file)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rectangle = (0, 56, 256, 150)
    
        mask = np.zeros(image_rgb.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)


        cv2.grabCut(image_rgb, # 원본 이미지
                   mask,       # 마스크
                   rectangle,  # 사각형
                   bgdModel,   # 배경을 위한 임시 배열
                   fgdModel,   # 전경을 위한 임시 배열 
                   5,          # 반복 횟수
                   cv2.GC_INIT_WITH_RECT)
        mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
        image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
        return image_rgb_nobg

    def makeSharp(width, height):
        image = cv2.imread('images/plane_256x256.jpg', cv2.IMREAD_GRAYSCALE)

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
 
        image_sharp = cv2.filter2D(image, -1, kernel)

        return image_sharp

    def crop(x0, y0, x1, y1):
        image = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE)
        image_cropped = image[y0:y1,x0:x1]

        return image_cropped
