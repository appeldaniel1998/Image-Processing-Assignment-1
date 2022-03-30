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
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)  # Converting to grayscale
    else:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting to RGB
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
    matrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]  # Constant matrix
    yiqMat = np.zeros_like(imgRGB.astype(float))  # creating an empty array of the correct type and size
    shape = imgRGB.shape[0]
    for val in range(shape):
        yiqMat[val, ...] = np.matmul(imgRGB[val, ...], matrix)  # every element in image where the value corresponds
    return yiqMat


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    matrix = [[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]  # Constant matrix
    rgbMat = np.zeros_like(imgYIQ.astype(float))  # creating an empty array of the correct type and size
    shape = rgbMat.shape[0]
    for val in range(shape):
        rgbMat[val, ...] = np.matmul(imgYIQ[val, ...], matrix)  # every element in image where the value corresponds
    return np.round(rgbMat, 10)  # round to 10 decimal places, to make up for double inaccuracy


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    showImg(imgOrig, imgOrig.ndim - 1)  # Present image
    cumHist, grayOrRGB = cumulativeHistogram(imgOrig)  # Returns a cumulative histogram (not normalized)
    lut = np.zeros(cumHist.shape)  # creating empty array
    shape = imgOrig.shape
    for i in range(len(cumHist)):  # Filling LUT
        lut[i] = math.ceil((cumHist[i] / (shape[0] * shape[1])) * 255)
    imgEqualized = np.zeros(imgOrig.shape)  # creating array to be returned, to be made the equalized image
    if grayOrRGB == "gray":  # Image is grayscale
        for i in range(len(imgEqualized)):  # Iterating over rows of image
            for j in range(len(imgEqualized[0])):  # Iterating over columns (elements of rows)
                imgEqualized[i][j] = lut[int(np.round(imgOrig[i][j] * 255))]
    else:  # Image is RGB
        imgEqualized = transformRGB2YIQ(imgOrig)  # Transforming RGB to YIQ
        imgEqualized[..., 0] = (imgEqualized[..., 0] - np.min(imgEqualized[..., 0])) / (np.max(imgEqualized[..., 0]) - np.min(imgEqualized[..., 0])) * 255  # Normalizing to 0-255
        for i in range(len(imgEqualized)):  # Iterating over rows of image
            for j in range(len(imgEqualized[0])):  # Iterating over columns (elements of rows)
                imgEqualized[i][j][0] = lut[int(np.round(imgEqualized[i][j][0]))]
        imgEqualized[..., 0] = imgEqualized[..., 0] / 255  # Normalizing to 0-1
        imgEqualized = transformYIQ2RGB(imgEqualized)  # Transforming back to RGB from YIQ

    histOrigin = histogramFromImg(imgOrig)[0]  # histogram of the original image
    equalizedNormalizedImg = normalize(imgEqualized)
    histEq = histogramFromImg(equalizedNormalizedImg)[0]  # Histogram of the equalized image

    showImg(imgEqualized, imgOrig.ndim - 1)

    return imgEqualized, histOrigin, histEq


def histogramFromImg(img) -> (np.ndarray, str):
    """
    :param img: normalized image (grayscale or RGB)
    :return: image histogram (regular histogram if grayscale, Y channel of YIQ if RGB image) and whether the image is RGB or grayscale
    """
    ret = np.zeros(256)  # creating an empty histogram
    if img.ndim == 2:  # Grayscale image
        unique, counts = np.unique(np.round(img * 255), return_counts=True)  # returns the unique values and how many of each value were found
        for i in range(len(counts)):
            ret[int(unique[i])] = counts[i]  # Filling the histogram
        return ret, "gray"
    else:  # RGB image
        img = transformRGB2YIQ(img)[..., 0]  # taking the Y channel of the YIQ image
        scaledImg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255  # scaling to 0-255
        unique, counts = np.unique(np.round(scaledImg), return_counts=True)  # getting the unique rounded values of the Y channel
        for i in range(len(counts)):
            ret[int(unique[i])] = counts[i]  # Filling the histogram
        return ret, "rgb"


def cumulativeHistogram(img) -> (np.ndarray, str):
    """Given an input of an image (as numpy array), calculates and returns the normalized cumulative histogram as numpy array"""
    hist, grayOrRGB = histogramFromImg(img)  # calling the function to get the regular histogram (Y channel in case of RGB images)
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
    sections = initialSections(nQuant, imOrig)  # Initializing sections
    retImages = [imOrig]  # initializing the returning array of images (adding the original image as the first element
    hist, imgType = histogramFromImg(retImages[-1])  # Getting the histogram of the image for optimal point choosing
    retErrors = []

    for i in range(1, nIter):
        optimalPoints = optimalPointsInSections(hist, sections)  # choosing optimal points for given sections
        sections = optimalSectionsGivenPixels(optimalPoints)  # setting optimal sections for given points
        newImg = histogramToImg(retImages[-1], imgType, sections, optimalPoints)  # applying colour changes to images
        retImages.append(newImg)  # Contains normalized RGB images
        retErrors.append(WMSEErrorCalc(hist, sections, optimalPoints, retImages[0]))
        try:  # If error doesnt change, stop the loop - no need for further analysis
            if retErrors[-1] == retErrors[-2]:
                break
        except IndexError:  # if first element, the other element doesn't exist. do nothing if so.
            pass

    plt.plot(retErrors)
    plt.show()
    if imgType == "gray":
        imgType = 1
    else:
        imgType = 2
    showImg(retImages[0], imgType)
    showImg(retImages[-1], imgType)
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


def optimalPointsInSections(histogram: np.ndarray, sections: np.ndarray) -> np.ndarray:
    optimalPoints = np.zeros(len(sections) - 1)
    for sectionInd in range(1, len(sections)):
        # Numerator + Denominator calculation according to the given formula
        numeratorIntegral = 0
        denominatorIntegral = 0
        for i in range(int(sections[sectionInd - 1]), int(sections[sectionInd])):  # for every element in a section
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
        sections[i] = (optimalPoints[i - 1] + optimalPoints[i]) / 2  # Average between the 2 points

    return sections


def histogramToImg(imOrig: np.ndarray, imgType: str, sections: np.ndarray, optimalPoints: np.ndarray) -> (np.ndarray, np.ndarray):
    imOrig = imOrig.copy()
    if imgType == "gray":
        imOrig *= 255
        for i in range(len(imOrig)):  # For each section and for each pixel, find all pixels in need of change, and change them appropriately to the correct value
            for j in range(len(imOrig[0])):
                for sectionId in range(1, len(sections)):
                    if sections[sectionId - 1] <= imOrig[i][j] <= sections[sectionId]:
                        imOrig[i][j] = optimalPoints[sectionId - 1]
                        break
        return imOrig / 255  # normalize back to 0-1
    else:
        # Received RGB normalized image
        yiqImage = transformRGB2YIQ(imOrig)  # transform to YUQ
        yChannel = yiqImage[:, :, 0].ravel()  # take the Y channel and flatten the array for easier work
        yChannel = np.round((yChannel - np.min(yChannel)) / (np.max(yChannel) - np.min(yChannel)) * 255)
        for yValIndex in range(len(yChannel)):  # For each section and for each pixel, find all pixels in need of change, and change them appropriately to the correct value
            for sectionIndex in range(1, len(sections)):
                if sections[sectionIndex - 1] <= yChannel[yValIndex] <= sections[sectionIndex]:
                    yChannel[yValIndex] = optimalPoints[sectionIndex - 1]
        yChannel /= 255  # Normalizing to 0-1
        yChannel = yChannel.reshape((yiqImage.shape[0], yiqImage.shape[1]))  # reshaping to original shape
        yiqImage[:, :, 0] = yChannel
        rgbImg = transformYIQ2RGB(yiqImage)  # transforming back to RGB
        return rgbImg


def WMSEErrorCalc(hist: np.ndarray, sections: np.ndarray, optimalPoints: np.ndarray, origImg: np.ndarray) -> float:
    errorSum = 0
    for i in range(1, len(sections)):
        for j in range(int(sections[i - 1]), int(sections[i])):
            errorSum += (hist[j] / (origImg.shape[0] * origImg.shape[1])) * ((j - optimalPoints[i - 1]) ** 2)
    return errorSum
