import cv2
import numpy as np

PERSON_CLASS_ID = 0

# Load YOLO model for person detection
net_person = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load gender classification model
net_gender = cv2.dnn.readNet("gender_net.caffemodel", "deploy_gender.prototxt")

# Load an image
image = cv2.imread("works.jpg")

# Get the image's spatial dimensions
height, width = image.shape[:2]

# Prepare the image for YOLO input
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input for the person detection network
net_person.setInput(blob)

# Get the output layer names
output_layer_names = net_person.getUnconnectedOutLayersNames()

# Run forward pass for person detection
outputs = net_person.forward(output_layer_names)

# Initialize lists for detected objects' information
boxes = []
confidences = []

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

# Labels for gender classification
gender_labels = ["Male", "Female"]

# Process each detected person
for i in indices:
    x, y, w, h = boxes[i]
    person_roi = image[y : y + h, x : x + w]

    # Preprocess the image for gender classification
    gender_blob = cv2.dnn.blobFromImage(
        person_roi,
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
        crop=False,
    )
    net_gender.setInput(gender_blob)

    # Run forward pass for gender classification
    gender_output = net_gender.forward()

    # Get the gender prediction
    gender_class_id = np.argmax(gender_output)
    gender_prediction = gender_labels[gender_class_id]

    # Draw rectangle around face and display gender label
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    y_text = y - 10 if y - 10 > 10 else y + 10
    cv2.putText(
        image,
        gender_prediction,
        (x, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    if gender_prediction == "Female":
        blurred_roi = cv2.GaussianBlur(person_roi, (99, 99), 30)
        image[y : y + h, x : x + w] = blurred_roi

# Display the result
cv2.imshow("Gender Person Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
