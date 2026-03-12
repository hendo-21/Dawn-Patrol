import os
import json
import datetime
import numpy as np
import cv2 as cv
import time
from dotenv import load_dotenv
from capture import capture_frame
from detect import detect_surfers, create_mask


load_dotenv()
STREAM_URL = os.getenv('STREAM_URL')
# Read mask file to exclude image area
with open('frames/raw/new_mask.json', 'r') as file:
    data = json.load(file)

# Get the polygon points and convert to numpy array for use in fillPoly()
# Open CV method fillPoly() precondition dtype must be signed 32bit int
points = data['shapes'][0]['points']
EXCLUDED_AREA = np.array(points, dtype=np.int32)


def main():
    # Capture 1 frame every 10 seconds
    frame_count = 0
    counts = []
    while frame_count < 3:
        counts.append(capture_and_count())
        frame_count += 1
        time.sleep(10)
    avg = int(sum(counts) / frame_count)
    return avg


def capture_and_count():
    frame = capture_frame(STREAM_URL)
    count = 0
    if frame is not None:
        masked_frame = create_mask(frame, EXCLUDED_AREA)
        cv.imwrite('frames/test_frame.jpg', masked_frame)
        count = detect_surfers(masked_frame, EXCLUDED_AREA)
    return count


if __name__ == '__main__':
    main()
