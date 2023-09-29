from typing import Literal

import cv2
import numpy as np

Gender = Literal["male", "female"]

# Load gender classification model
net_gender = cv2.dnn.readNet(
    "src/models/gender_prediction/gender_net.caffemodel",
    "src/models/gender_prediction/deploy_gender.prototxt",
)


def get_gender(roi: np.ndarray) -> Gender:
    """
    Get a ROI (Region Of Interest) and returns either the person is a male or a female

    NOTE: not 100% accurate
    """
    # Preprocess the image for gender classification
    gender_blob = cv2.dnn.blobFromImage(
        image=roi,
        scalefactor=1.0,
        size=(227, 227),
        mean=(78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
        crop=False,
    )
    net_gender.setInput(blob=gender_blob)

    # Run forward pass for gender classification
    gender_output = net_gender.forward()

    # Get the gender prediction
    gender_class_id = np.argmax(gender_output)

    return "male" if gender_class_id == 0 else "female"
