import time

import cv2

from hafdh_nadhar.blur_person import blur_persons
from hafdh_nadhar.person_detection import get_person_boxes


def main(img_path: str):
    for i in range(5):
        print(i)
        # Mesure du temps avant de charger l'image
        start_time = time.time()

        # Load an image
        image = cv2.imread(filename=img_path)
        if image is None:
            raise FileNotFoundError(f"No valid image found here: {img_path}")

        # Calcul du temps écoulé pour charger l'image
        load_time = time.time() - start_time
        print(f"Image loaded in {load_time:.2f} seconds")

        # Mesure du temps avant de détecter les boîtes de personne
        start_time = time.time()

        person_boxes = get_person_boxes(image=image)

        # Calcul du temps écoulé pour détecter les boîtes de personne
        detection_time = time.time() - start_time
        print(f"Person boxes detected in {detection_time:.2f} seconds")

        # Mesure du temps avant de flouter les personnes
        start_time = time.time()

        blur_persons(image=image, person_boxes=person_boxes, only_female=False)

        # Calcul du temps écoulé pour flouter les personnes
        blur_time = time.time() - start_time
        print(f"Persons blurred in {blur_time:.2f} seconds")

        # Display the result
        cv2.imshow("Result", image)
        print("Result displayed")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
