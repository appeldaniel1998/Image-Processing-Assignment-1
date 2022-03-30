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

from ex1_utils import *


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    def nothing(x):
        pass

    imgArr = imReadAndConvert(img_path, rep)
    title = "Gamma Correction"
    cv2.namedWindow(title)
    trackbar_name = "Gamma Variable"
    cv2.createTrackbar(trackbar_name, title, 100, 200, nothing)

    while cv2.getWindowProperty(title, 0) >= 0:  # the while loop continues until the window is closed or the esc key is pressed
        trackVal = cv2.getTrackbarPos(trackbar_name, title)
        newImg = imgArr ** (trackVal / 100)  # applying gamma correction
        cv2.imshow(title, newImg)

        key = cv2.waitKey(1)
        if key == 27:  # esc key is pressed
            break
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
