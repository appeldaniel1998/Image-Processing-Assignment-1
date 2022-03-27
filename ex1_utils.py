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
import math
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
    else:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return normalize(img)


def normalize(img: np.ndarray) -> np.ndarray:
    maximum_value, minimum_value = np.max(img), np.min(img)
    normalizedArr = (img - float(minimum_value)) / float(maximum_value - minimum_value)
    return normalizedArr


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img_array = imReadAndConvert(filename, representation)
    showImg(img_array, representation)


def showImg(img_array: np.ndarray, representation: int):
    if representation == 1:
        plt.imshow(img_array, cmap='gray')
        plt.show()
        # cv2.imshow("", img_array)
        # cv2.waitKey(0)
    else:
        if representation == 2:
            plt.imshow(img_array)
            plt.show()
            # cv2.imshow("", img_array)
            # cv2.waitKey(0)


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
    return np.round(rgbMat, 10)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    showImg(imgOrig, imgOrig.ndim - 1)
    cumHist, grayOrRGB = cumulativeHistogram(imgOrig)
    lut = np.zeros(cumHist.shape)
    shape = imgOrig.shape
    for i in range(len(cumHist)):  # Filling LUT
        lut[i] = math.ceil((cumHist[i] / (shape[0] * shape[1])) * 255)
    imgEqualized = np.zeros(imgOrig.shape)
    if grayOrRGB == "gray":  # Image is grayscale
        for i in range(len(imgEqualized)):  # Iterating over rows of image
            for j in range(len(imgEqualized[0])):  # Iterating over columns (elements of rows)
                imgEqualized[i][j] = lut[int(np.round(imgOrig[i][j] * 255))]
    else:  # Image is RGB
        imgEqualized = transformRGB2YIQ(imgOrig)
        imgEqualized[..., 0] = (imgEqualized[..., 0] - np.min(imgEqualized[..., 0])) / (np.max(imgEqualized[..., 0]) - np.min(imgEqualized[..., 0])) * 255
        for i in range(len(imgEqualized)):  # Iterating over rows of image
            for j in range(len(imgEqualized[0])):  # Iterating over columns (elements of rows)
                imgEqualized[i][j][0] = lut[int(np.round(imgEqualized[i][j][0]))]
        imgEqualized[..., 0] = imgEqualized[..., 0] / 255
        imgEqualized = transformYIQ2RGB(imgEqualized)

    histOrigin = histogramFromImg(imgOrig)[0]
    equalizedNormalizedImg = normalize(imgEqualized)
    histEq = histogramFromImg(equalizedNormalizedImg)[0]

    showImg(imgEqualized, imgOrig.ndim - 1)

    return imgEqualized, histOrigin, histEq


def histogramFromImg(img) -> (np.ndarray, str):
    """
    :param img: normalized image (grayscale or RGB)
    :return: image histogram (regular histogram if grayscale, Y channel of YIQ if RGB image) and whether the image is RGB or grayscale
    """
    ret = np.zeros(256)
    if img.ndim == 2:  # Grayscale image
        unique, counts = np.unique(np.round(img * 255), return_counts=True)
        for i in range(len(counts)):
            ret[int(unique[i])] = counts[i]
        return ret, "gray"
    else:  # RGB image
        img = transformRGB2YIQ(img)[..., 0]
        scaler = MinMaxScaler(feature_range=(0, 255))
        scaledImg = scaler.fit_transform(img)
        unique, counts = np.unique(np.round(scaledImg), return_counts=True)
        for i in range(len(counts)):
            ret[int(unique[i])] = counts[i]
        return ret, "rgb"


def cumulativeHistogram(img) -> (np.ndarray, str):
    """Given an input of an image (as numpy array), calculates and returns the normalized cumulative histogram as numpy array"""
    hist, grayOrRGB = histogramFromImg(img)
    cumHist = np.cumsum(hist)
    return cumHist, grayOrRGB


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    sections = initialSections(nQuant, imOrig)
    optimalPoints = initializePoints(sections)
    retImages = [imOrig]
    retErrors = []

    for i in range(1, nIter):
        hist, imgType = histogramFromImg(retImages[-1])
        optimalPoints = optimalPointsInSections(hist, sections)
        sections = optimalSectionsGivenPixels(optimalPoints)
        retImages.append(histogramToImg(retImages[-1], imgType, sections, optimalPoints, hist))
        retErrors.append(WMSEErrorCalc(hist, sections, optimalPoints))
    plt.plot(retErrors)
    return retImages, retErrors


def initialSections(nQuant: int, imOrig: np.ndarray) -> np.ndarray:
    """
    Initialize the sections to values such that there are approximately the same number of pixels in each section
    :param nQuant: number of dividing lines on the histogram
    :param imOrig: Original image as numpy array
    :return: the sections initialized to the correct values (currently correct)
    """
    hist = histogramFromImg(imOrig)[0]
    sections = np.zeros(nQuant + 1)

    # Constants
    sections[0] = 0
    sections[-1] = 255

    approxNumOfPixelsInSections = (imOrig.shape[0] * imOrig.shape[1]) / (nQuant + 1)
    index = 0
    for i in range(1, len(sections) - 1):
        tempSum = 0
        while tempSum < approxNumOfPixelsInSections:
            tempSum += hist[index]
            index += 1
        sections[i] = index - 1
    return sections


def initializePoints(sections: np.ndarray) -> np.ndarray:
    optimalPoints = np.zeros(len(sections) - 1)
    for i in range(1, len(sections)):
        optimalPoints[i - 1] = int(np.round((sections[i] + sections[i - 1]) / 2))
    return optimalPoints


def optimalPointsInSections(histogram: np.ndarray, sections: np.ndarray) -> np.ndarray:
    optimalPoints = np.zeros(len(sections) - 1)
    for sectionInd in range(1, len(sections)):
        # Numerator + Denominator calculation
        numeratorIntegral = 0
        denominatorIntegral = 0
        for i in range(int(sections[sectionInd - 1]), int(sections[sectionInd])):
            numeratorIntegral += histogram[i] * i
            denominatorIntegral += histogram[i]

        optimalPoints[sectionInd - 1] = np.round(numeratorIntegral / denominatorIntegral)
    return optimalPoints


def optimalSectionsGivenPixels(optimalPoints: np.ndarray) -> np.ndarray:
    sections = np.zeros(len(optimalPoints) + 1)

    # Constants
    sections[0] = 0
    sections[-1] = 255

    for i in range(1, len(sections) - 1):
        sections[i] = (optimalPoints[i - 1] + optimalPoints[i]) / 2

    return sections


def histogramToImg(imOrig: np.ndarray, imgType: str, sections: np.ndarray, optimalPoints: np.ndarray, hist: np.ndarray) -> np.ndarray:
    imOrig = imOrig.copy()
    if imgType == "gray":
        imOrig *= 255
        for i in range(len(imOrig)):
            for j in range(len(imOrig[0])):
                for sectionId in range(1, len(sections)):
                    if sections[sectionId - 1] <= imOrig[i][j] <= sections[sectionId]:
                        imOrig[i][j] = optimalPoints[sectionId - 1]
                        break
        return imOrig / 255
    else:  # RGB Image
        yiqImg = transformRGB2YIQ(imOrig)
        yiqImg[..., 0] = (yiqImg[..., 0] - np.min(yiqImg[..., 0])) / (np.max(yiqImg[..., 0]) - np.min(yiqImg[..., 0])) * 255
        YChannel = yiqImg[..., 0]
        for secInd in range(1, len(sections)):
            YChannel[sections[secInd - 1] <= YChannel <= sections[secInd]] = optimalPoints[secInd - 1]
        yiqImg[..., 0] = YChannel
        yiqImg[..., 0] /= 255
        return transformYIQ2RGB(yiqImg)


def WMSEErrorCalc(hist: np.ndarray, sections: np.ndarray, optimalPoints: np.ndarray) -> float:
    errorSum = 0
    for i in range(1, len(sections)):
        for j in range(int(sections[i - 1]), int(sections[i])):
            errorSum += hist[j] * ((j - optimalPoints[i - 1]) ** 2)
    return errorSum







# imOrig = transformRGB2YIQ(imOrig)
# # imOrig[..., 0] = np.round(imOrig[..., 0] * 255)
# scaler = MinMaxScaler(feature_range=(0, 255))
# imOrig[..., 0] = scaler.fit_transform(imOrig[..., 0])
# imOrig = np.round(imOrig[..., 0])
# for i in range(len(imOrig)):
#     for j in range(len(imOrig[0])):
#         for sectionId in range(1, len(sections)):
#             if sections[sectionId - 1] <= imOrig[i][j][0] <= sections[sectionId]:
#                 imOrig[i][j][0] = optimalPoints[sectionId - 1]
#                 break
# imOrig[..., 0] /= 255
# imOrig = transformYIQ2RGB(imOrig)

# imgEqualized = transformRGB2YIQ(imOrig)
# imgEqualized[..., 0] = (imgEqualized[..., 0] - np.min(imgEqualized[..., 0])) / (np.max(imgEqualized[..., 0]) - np.min(imgEqualized[..., 0])) * 255
# for i in range(len(imgEqualized)):  # Iterating over rows of image
#     for j in range(len(imgEqualized[0])):  # Iterating over columns (elements of rows)
#         for sectionId in range(1, len(sections)):
#             if sections[sectionId - 1] <= imgEqualized[i][j][0] <= sections[sectionId]:
#                 imgEqualized[i][j][0] = optimalPoints[sectionId - 1]
# imgEqualized[..., 0] = imgEqualized[..., 0] / 255
# imgEqualized = transformYIQ2RGB(imgEqualized)
