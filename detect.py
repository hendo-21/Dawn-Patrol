"""
This module provides support for surfer detection in a frame.
creates a mask...
count returns a count of surfers in the frame
a_frame returns an annotated frame for debugging
"""

import numpy as np
import cv2 as cv
import datetime

from config import MODEL_CONFIDENCE


def create_mask(image, excluded_area):
    mask = np.zeros_like(image)

    cv.fillPoly(mask, [excluded_area], (255, 255, 255))
    framed = cv.bitwise_and(image, mask)

    return framed


def detect_surfers(frame, excluded_area, model, conf=MODEL_CONFIDENCE):
    file_timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    # Write image to disk (Roboflow API precondition)
    masked_path = f"frames/raw/masked_{file_timestamp}.jpg"
    cv.imwrite(masked_path, frame)

    result = model.predict(masked_path, confidence=conf)
    predictions = result.json()["predictions"]

    # Annotate frame with prediction boxes for debugging
    result.save(f"frames/annotated/annotated_frame{file_timestamp}.jpg")

    surfer_ct = 0
    for pred in predictions:
        center_point = (int(pred["x"]), int(pred["y"]))
        if (
            pred["class"] == "surfer"
            and cv.pointPolygonTest(excluded_area, center_point, False) >= 0
        ):
            surfer_ct += 1

    return surfer_ct
