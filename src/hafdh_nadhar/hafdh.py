from typing import TYPE_CHECKING, Optional

import cv2

from hafdh_nadhar.blur_person import blur_persons
from hafdh_nadhar.helpers import save_image, show_image
from hafdh_nadhar.person_detection import get_person_boxes

if TYPE_CHECKING:
    import numpy as np


def hafdh_img(
    img_path: str, open_in_window: bool = False, save_to_path: Optional[str] = None
) -> "np.ndarray":
    """
    Hafdh (read: حفظ | means: preserve) an image from human representations by blurring them

    Args:
        img_path (str): the path to the image
        open_window (bool, optional): whether to open a window to display the result. Defaults to False.
        save_to_path (Optional[str], optional): the path to save the image to. Defaults to None.

    Returns:
        np.ndarray: the image after hafdh
    """
    # Load an image
    image = cv2.imread(filename=img_path)
    if image is None:
        raise FileNotFoundError(f"No valid image found here: {img_path}")

    person_boxes = get_person_boxes(image=image)
    blur_persons(image=image, person_boxes=person_boxes, only_female=False)

    if open_in_window:
        show_image(image=image)

    if save_to_path:
        save_image(image=image, img_path=save_to_path)

    return image
