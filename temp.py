from ex1_utils import *


def main():

    img = cv2.cvtColor(cv2.imread("beach.jpg"), cv2.COLOR_BGR2GRAY)
    # histogram = np.zeros((256, img.ndim))
    if img.ndim == 2:
        unique, counts = np.unique(img, return_counts=True)
        histogram = np.array(counts)
    print(unique.shape)

if __name__ == '__main__':
    main()
