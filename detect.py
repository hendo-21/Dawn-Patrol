"""
This module provides support for surfer detection in a frame.
creates a mask...
count returns a count of surfers in the frame
a_frame returns an annotated frame for debugging
"""
import numpy as np
import cv2 as cv
import datetime
import json
from ultralytics import YOLO

PERSON_CLASS_ID = 0
IMG_PATH = 'frames/raw/frame_260306-1630.jpg'
FILE_TIMESTAMP = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

# Define the YOLO model and perform inference
model = YOLO('yolo26n.pt')
MODEL_CONF = 0.1

# Load the mask file
with open('frames/raw/mask.json', 'r') as file:
    data = json.load(file)

# Get the points and build the ndarray
points = data['shapes'][0]['points']
# Open CV method fillPoly() precondition dtype must be signed 32bit int
POLY_MASK = np.array(points, dtype=np.int32)


def get_frame():
    frame = cv.imread(IMG_PATH)
    if frame is None:
        return None
    return frame


def create_mask(image: np.ndarray):
    # Fill image area with black
    mask = np.zeros_like(image)
    cv.fillPoly(mask, [POLY_MASK], (255, 255, 255))
    framed = cv.bitwise_and(image, mask)
    cv.imwrite('frames/masked.jpg', framed)
    return framed


def detect_surfers(frame: np.ndarray, roi_mask: np.ndarray | None = None):
    results = model.predict(frame, conf=MODEL_CONF)

    # Save the annotated frame
    annotated_frame = results[0].plot()
    cv.imwrite(
        f'frames/annotated/annotated_frame{FILE_TIMESTAMP}.jpg',
        annotated_frame
    )

    # Isoloate the class id tensor and count class id person
    surfer_ct = 0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if int(box.cls) == PERSON_CLASS_ID:
                    surfer_ct += 1
    return surfer_ct


if __name__ == '__main__':
    img = get_frame()
    edited = create_mask(img)
    if img is not None:
        r = detect_surfers(edited)
        print(f'Surfer count: {r}')
