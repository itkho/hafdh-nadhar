import cv2

from hafdh_nadhar.blur_person import blur_persons
from hafdh_nadhar.person_detection import get_person_boxes


def hafdh_img(img_path: str):
    """Hafdh (read: حفظ | means: preserve) an image from human representations by blurring them"""
    # Load an image
    image = cv2.imread(filename=img_path)
    if image is None:
        raise FileNotFoundError(f"No valid image found here: {img_path}")

    person_boxes = get_person_boxes(image=image)
    blur_persons(image=image, person_boxes=person_boxes, only_female=False)

    # Display the result
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
