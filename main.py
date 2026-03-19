import os
import json
import numpy as np
import cv2 as cv
import time
from dotenv import load_dotenv
from capture import capture_frame
from detect import detect_surfers, create_mask
from notify import send_notification


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
    avg_surfer_count = aggregate_surfer_counts()
    send_notification(
        avg_surfer_count,
        os.getenv("PUSHCUT_API_KEY"),
        os.getenv("PUSHCUT_URL")
    )


def capture_and_count():
    # Return None if .env not setup with URL
    if STREAM_URL is None:
        print("Could not fetch stream URL.")
        return None

    # Capture a frame
    frame = capture_frame(STREAM_URL)

    # Return None if there's an issue with frame capture.
    if frame is None:
        return None

    # Mask the frame to define region of interest, then run surfer detection
    count = 0
    if frame is not None:
        masked_frame = create_mask(frame, EXCLUDED_AREA)
        count = detect_surfers(masked_frame, EXCLUDED_AREA)
    return count


def aggregate_surfer_counts():
    # Capture 1 frame every 10 seconds
    frame_count = 0
    counts = []
    while frame_count < 1:
        # Return None if there's an issue with frame capture
        result = capture_and_count()
        if result is None:
            return None

        # Append count to counts arr and then wait 10s before repeating
        counts.append(capture_and_count())
        frame_count += 1
        time.sleep(10)

    # Return average count of surfers
    return int(sum(counts) / frame_count)


if __name__ == '__main__':
    main()
