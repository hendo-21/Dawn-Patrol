"""
This module provides support for surfer detection in a frame.
creates a mask...
count returns a count of surfers in the frame
a_frame returns an annotated frame for debugging
"""
import numpy as np
import cv2 as cv
import datetime
from ultralytics import YOLO

# Config
PERSON_CLASS_ID = 0

# Define the YOLO model, inference classes and confidence threshold
model = YOLO('yolo26n.pt')
# model = YOLOE('yoloe-26l-seg.pt')
# model.set_classes(["person"])
MODEL_CONF = 0.075


def create_mask(image, excluded_area):
    # Fill image area with black
    mask = np.zeros_like(image)

    # Create a mask using the polygon and color it white
    cv.fillPoly(mask, [excluded_area], (255, 255, 255))
    framed = cv.bitwise_and(image, mask)

    # Optimization
    # x, y, w, h = cv.boundingRect(EXCLUDED_AREA)
    # cropped = framed[y:y+h, x:x+w]
    return framed


def detect_surfers(frame, excluded_area, conf=MODEL_CONF):
    file_timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    results = model.predict(frame, conf=conf)

    # Save the annotated frame
    annotated_frame = results[0].plot()
    cv.imwrite(
        f'frames/annotated/annotated_frame{file_timestamp}.jpg',
        annotated_frame
    )

    # Isoloate the class id tensor and count class id person
    surfer_ct = 0
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Extract the center point
                x, y = box.xywh[0][0], box.xywh[0][1]

                # Convert tensor values to float tuple: pointPolygonTest precon
                center_point = (float(x), float(y))
                if cv.pointPolygonTest(excluded_area, center_point, False) >= 0 and (int(box.cls) == PERSON_CLASS_ID or int(box.cls) == 14):
                    surfer_ct += 1
    return surfer_ct
