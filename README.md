# Dawn Patrol - In Progress

## Demo
Coming soon...

## Description
Daily text messages on surf conditions at my favorite local surf spot. Uses OpenCV to capture images from public beach camera footage, and YOLO object detection for image analysis.

## Context
I wanted to get my hands dirty with some of the open source image capture and object detection models/libraries. My local surf spot can be incredibly crowded, making the 45 minutes of traffic unappealing. This project gives me daily updates on surfer count so I don't have to check local feeds.

## Tech Stack
YOLO26, OpenCV

## Features
- Capture surf break footage from public beach cam video feed

## Project Structure
```
dawn-patrol/
├── frames/          # storage for video frames including raw and annotated captures
├── main.py          # entry point / scheduler
├── capture.py       # capture and return a frame from webcame
├── detect.py        # mask creation, YOLO inference, object counting
├── notify.py        # SMS logic
└── store.py         # logging counts to JSON/SQLite
```

## What I've Learned So Far
- Hosted video formats (HLS vs MJPEG)
- Practical CV tooling (OpenCV, Ultralytics, labelme)
- Why naive approaches fail on real-world scenes
- How object detection models work at a conceptual level
- Tradeoffs between custom model training and using a mask to reduce noise and improve detection

## Future Improvements
Wave analysis using inference.

## License
[MIT](LICENSE.md)
