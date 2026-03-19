"""
This is a script for testing model confidence threshold. The global
dictionary IMG_PATHS stores image paths and actual hand counted
number of surfers in the image.

Note, change file path for annotated frames to 'frames/conf_testing'
if you want to view the annotated frames at each threshold.
"""

from detect import create_mask, detect_surfers
from ultralytics import YOLO
from dotenv import load_dotenv
import cv2 as cv
import json
import numpy as np

load_dotenv()

# Load the file with the mask created from Labelme
with open('frames/raw/new_mask.json', 'r') as file:
    data = json.load(file)

# Get the polygon points and convert to numpy array for use in fillPoly()
# Open CV method fillPoly() precondition dtype must be signed 32bit int
points = data['shapes'][0]['points']
EXCLUDED_AREA = np.array(points, dtype=np.int32)

IMG_PATHS = {
    "frames/raw/frame_260319-130218.jpg": 47,
    "frames/raw/frame_260319-125958.jpg": 43,
    "frames/raw/frame_260319-123524.jpg": 40,
    "frames/raw/frame_260319-122743.jpg": 48,
    "frames/raw/frame_260306-1630.jpg": 4,
    "frames/raw/frame_260306-1531.jpg": 1
}
model = YOLO('yolo26n.pt')


def main():
    for file in IMG_PATHS:
        frame = cv.imread(file)
        masked_frame = create_mask(frame, EXCLUDED_AREA)
        result = test_confidence_thresholds(masked_frame)
        print_results(file, result)


def test_confidence_thresholds(masked_frame):
    result = {0.03: 0, 0.05: 0, 0.075: 0, 0.1: 0, 0.15: 0, 0.2: 0}

    # Run inference on the frame at each confidence level
    for conf_level in result:
        result[conf_level] = detect_surfers(
            masked_frame,
            EXCLUDED_AREA,
            conf_level
        )
    return result


def print_results(file_path, results):
    actual_count = IMG_PATHS[file_path]

    # Print the results
    print("\nFile path:", file_path)
    print("-- Surfers Detected per Confidence Level --\n")
    print("Conf      Count    Delta")
    for conf_level in results:
        delta = results[conf_level] - actual_count
        print(f"{conf_level:<5.3f}     {results[conf_level]:<2}       {delta:+}")


if __name__ == "__main__":
    main()
