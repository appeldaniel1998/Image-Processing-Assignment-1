"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from) - I would never:)
    :return: int
    """
    return 207386699


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        normalizedArr = preprocessing.normalize(img)
    else:

        img = cv2.imread("beach.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normalizedArr = np.zeros_like(img.astype(float))
        shape = img.shape[0]
        maximum_value, minimum_value = img.max(), img.min()

        # Normalize all the pixel values of the img to be from 0 to 1
        for val in range(shape):
            normalizedArr[val, ...] = (img[val, ...] - float(minimum_value)) / float(maximum_value - minimum_value)

    return normalizedArr


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img_array = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(img_array, cmap='gray')
        plt.show()
    else:
        if representation == 2:
            plt.imshow(img_array)
            plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    matrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]
    yiqMat = np.zeros_like(imgRGB.astype(float))
    shape = imgRGB.shape[0]
    for val in range(shape):
        yiqMat[val, ...] = np.matmul(imgRGB[val, ...], matrix)
    return yiqMat


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    matrix = [[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]
    rgbMat = np.zeros_like(imgYIQ.astype(float))
    shape = rgbMat.shape[0]
    for val in range(shape):
        rgbMat[val, ...] = np.matmul(imgYIQ[val, ...], matrix)
    return rgbMat


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
