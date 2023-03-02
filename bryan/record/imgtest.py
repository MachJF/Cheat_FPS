from PIL import Image, ImageGrab
import cv2
import numpy as np
import time


def img2mat(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


if __name__ == '__main__':
    im = ImageGrab.grab()
    cv2.imshow('frame', img2mat(im))
