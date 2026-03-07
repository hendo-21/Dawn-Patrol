"""
This module provides support for surfer detection in a frame.
creates a mask...
count returns a count of surfers in the frame
a_frame returns an annotated frame for debugging
"""

# Packages
import numpy as np
import cv2 as cv
from ultralytics import YOLO

PERSON_CLASS_ID = 0

# Define the YOLO model and perform inference
model = YOLO('yolo26n.pt')
results = model.predict('frames/raw/frame_260306-1531.jpg', conf=0.1)

# Save the annotated frame
annotated_frame = results[0].plot()
cv.imwrite('frames/annotated/annotated_frame.jpg', annotated_frame)


def get_frame(image_path):
    frame = cv.imread(image_path)
    if frame is None:
        return None
    return frame


def create_mask():
    pass


def detect_surfers(frame: np.ndarray, roi_mask: np.ndarray | None = None):
    pass
