# Image Prosessing: Assignment 1

The following project represents the first assignment in the course of Image Processing and Computer Vision.

The system used to run the task was Windows 10 PC, using PyCharm on python 3.8.

The files submitted are identical to the files given in the assignment material besides the two added test images. The changes were made in ex1_utils.py and gamma.py. (as well as the readme)

The files are as follows:
- gitignore - files to be ignored by the github, irrelevant to project
- Ex1.pdf - The pdf file of the assignment
- Readme.md - this file
- *.jpg, *.png - all test images used
- ex1_main - the main class provided in the assignment and was used for testing
- ex1_utils - the file where most major functions requested by the assignment have been implemented
- gamma.py - the file where the gamma correction function was implemented

The functions implemented in the asssignment (not the ones received as part of testing) are as follows:
- myID() - defines the ID number of the student handing in the assignment (me)
- imReadAndConvert() - Reads an image, and returns the image converted as requested (in grayscale or RGB)
- normalize() - normalizing the received array to the range [0-1]
- imDisplay() - Reads an image as RGB or GRAY_SCALE and displays it (in grayscale or RGB)
- showImg() - show image given an array representing it
- transformRGB2YIQ() - Converts an RGB image to YIQ color space
- transformYIQ2RGB() - Converts an YIQ image to RGB color space
- hsitogramEqualize() - Equalizes the histogram of an image
- histogramFromImg() - Calculating and returning the histogram of the image
- cumulativeHistogram() - Calculating and returning the cumulative histogram of the image
- quantizeImage() - Quantized an image in to nQuant colors
- initialSections() - Initialize the sections to values such that there are approximately the same number of pixels in each section
- optimalPointsInSections() - Choosing optimal points with the given sections
- optimalSectionsGivenPixels() - Choosing optimal sections with the given points
- histogramToImg() - applying quantization and converting to RGB (if RGB image)
- WMSEErrorCalc() - calculting weighted MSE given the quantized and the original images
- gammaDisplay() - GUI for gamma correction
- nothing() - does nothing
