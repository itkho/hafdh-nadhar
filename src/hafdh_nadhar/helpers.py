from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


def show_image(image: "np.ndarray"):
    """
    Show an image in a window and wait for a key press to exit
    """
    cv2.imshow("Image", image)
    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Note: don't know why, but it works (https://stackoverflow.com/a/63519593)
    cv2.waitKey(1)


def save_image(image: "np.ndarray", img_path: str):
    """
    Save an image to a file
    """
    cv2.imwrite(img_path, image)
