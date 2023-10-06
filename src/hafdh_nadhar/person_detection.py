import os

import cv2
import numpy as np

PERSON_CLASS_ID = 0

# Load YOLO model for person detection
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(os.path.relpath(__file__))
net_person = cv2.dnn.readNet(
    f"{current_dir}/models/person_prediction/yolov3-tiny.weights",
    f"{current_dir}/models/person_prediction/yolov3-tiny.cfg",
)


def get_person_boxes(image: np.ndarray) -> list[list[int]]:
    """
    Get an image and returns a list of boxes for each detected person

    Each box is a list of 4 elements: [x, y, w, h]

    x: the x coordinate of the top-left corner of the box
    y: the y coordinate of the top-left corner of the box
    w: the width of the box
    h: the height of the box
    """

    # Initialize lists for detected objects' information
    boxes = []
    confidences = []

    # Get the image's spatial dimensions
    height, width = image.shape[:2]

    # Prepare the image for YOLO input
    blob = cv2.dnn.blobFromImage(
        image=image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False
    )

    # Set the input for the person detection network
    net_person.setInput(blob)

    # Get the output layer names
    output_layer_names = net_person.getUnconnectedOutLayersNames()

    # Run forward pass for person detection
    outputs = net_person.forward(output_layer_names)

    # Process each output layer for person detection
    for output in outputs:
        for detection in output:
            # After the 4 first elements (which are for the box), we have the score for each class
            scores = detection[5:]

            # Check that the detected object is a person
            class_id = np.argmax(scores)
            if not class_id == PERSON_CLASS_ID:
                continue

            # Check that the score is highly enough
            confidence = scores[class_id]
            if confidence < 0.5:
                continue

            box = detection[0:4] * np.array([width, height, width, height])
            centerX, centerY, w, h = box.astype("int")

            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))

            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return [boxes[i] for i in indices]
