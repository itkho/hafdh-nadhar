from typing import TYPE_CHECKING

import cv2

from hafdh_nadhar.gender_detection import get_gender

if TYPE_CHECKING:
    import numpy as np


def blur_persons(
    image: "np.ndarray", person_boxes: list[list[int]], only_female: bool = False
):
    """
    Get an image and a list of boxes for each detected person and blur each person
    """
    for boxe in person_boxes:
        x, y, w, h = boxe
        person_roi = image[y : y + h, x : x + w]

        if only_female:
            gender_prediction = get_gender(roi=person_roi)
            if gender_prediction != "female":
                continue

        # Extend the person_roi
        extention = 40
        extended_person_roi = image[
            y - extention : y + h + extention, x - extention : x + w + extention
        ]

        # Blur the person
        blurred_roi = cv2.GaussianBlur(
            src=extended_person_roi, ksize=(99, 99), sigmaX=50
        )
        image[
            y - extention : y + h + extention, x - extention : x + w + extention
        ] = blurred_roi
