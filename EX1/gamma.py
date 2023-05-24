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
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import cv2

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = cv2.imread(img_path)

    if rep == 2:
        img = cv2.cvtColor(img_path)
    adjusted = adjust_gamma(img, gamma=100/100.0)
    # Create a window to display the image and trackbar
    cv2.namedWindow('Gamma Correction')
# Create a trackbar for adjusting the gamma value
    cv2.createTrackbar("Trackbar", 'Gamma Correction', 0, 100, on_trackbar_change)

    cv2.imshow('Gamma Corrected Image', adjusted)
    while True:
    # Do some processing here
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

def on_trackbar_change(value):
    print("Trackbar value:", value)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Create a trackbar for adjusting the gamma value
# cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 300, on_gamma_trackbar)
def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
