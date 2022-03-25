import numpy as np

from ex1_utils import *


def main():
    img = imReadAndConvert("beach.jpg", 2)
    hsitogramEqualize(img)


if __name__ == '__main__':
    main()
