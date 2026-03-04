"""
This module provides a function for capturing a single frame
of video from a URL stream.
"""

# Packages
import cv2 as cv
import numpy as np
import datetime


def capture_frame(stream_url: str) -> np.ndarray | None:
    """
    Captures a frame from the input URL and stores it.
    :param: stream_url - a string to a video feed
    :return: frame resolution and color channels as NumPy array
        or None if not captured.
    """
    # Get current datetime and format for filename
    file_timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M")

    # Capture a frame from the video
    capture = cv.VideoCapture(stream_url)
    try:
        if not capture.isOpened():
            print("Cannot open URL.")
            return None

        # Read the frame (read() ret a success flag and the data as a tuple)
        flag, frame = capture.read()
        if flag:
            # Save the frame
            cv.imwrite(f"frames/frame_{file_timestamp}.jpg", frame)
            print(f"✅ Frame read. NumPy array: {frame.shape}. "
                  f"File name: frame_{file_timestamp}.jpg")
            return frame
        else:
            print("❌ Failed to read frame.")
            return None
    finally:
        # Release the capture
        capture.release()
