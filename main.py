import os
import json
import numpy as np
import time

from roboflow import Roboflow
from dotenv import load_dotenv
from config import FRAME_CAP_COUNT, EXCLUSION_AREA_FILENAME
from capture import capture_frame
from detect import detect_surfers, create_mask


load_dotenv()
STREAM_URL = os.getenv("STREAM_URL")
# Read mask file to exclude image area
with open(f"frames/{EXCLUSION_AREA_FILENAME}", "r") as file:
    data = json.load(file)

# Get the polygon points and convert to numpy array for use in fillPoly()
# Open CV method fillPoly() precondition dtype must be signed 32bit int
points = data["shapes"][0]["points"]
EXCLUDED_AREA = np.array(points, dtype=np.int32)

# Load Roboflow trained model
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
workspace = rf.workspace(ROBOFLOW_WORKSPACE)
project = workspace.project(ROBOFLOW_PROJECT)
trained_models = project.version(2).models()
if not trained_models:
    raise RuntimeError("No trained models found for version 2 of this project.")
model = trained_models[0]


def main():
    avg_surfer_count = aggregate_surfer_counts()

    """
    send_notification(
        avg_surfer_count,
        os.getenv("PUSHCUT_API_KEY"),
        os.getenv("PUSHCUT_URL")
    )
    """

    print(avg_surfer_count)


def capture_and_count():
    # Return None if .env not setup with URL
    if not STREAM_URL:
        print("Could not fetch stream URL.")
        return None

    frame = capture_frame(STREAM_URL)
    if frame is None:
        return None

    # Mask the frame to define region of interest, then run surfer detection
    masked_frame = create_mask(frame, EXCLUDED_AREA)
    return detect_surfers(masked_frame, EXCLUDED_AREA, model)


def aggregate_surfer_counts():
    # Capture 1 frame every 10 seconds
    frame_count = 0
    counts = []
    while frame_count < FRAME_CAP_COUNT:
        # Return None if there's an issue with frame capture
        result = capture_and_count()
        if result is None:
            return None

        counts.append(result)
        frame_count += 1
        time.sleep(10)

    # Avg surfer count to reduce noise
    return int(sum(counts) / frame_count)


if __name__ == "__main__":
    main()
