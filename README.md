# Dawn Patrol - In Progress

## Demo
Coming soon...

## Description
Daily push notifications on surf conditions at my favorite local surf spot. Uses OpenCV to capture images from public beach camera footage, and YOLO object detection for image analysis.

## Context
I wanted to get my hands dirty with some of the open source image capture and object detection models/libraries. My local surf spot can be incredibly crowded, making the 45 minutes of traffic unappealing. This project gives me daily updates on surfer count so I don't have to check local feeds.

## Tech Stack
YOLO26, OpenCV, Pushcut

## Features
- Capture surf break footage from public beach cam video feed

## Project Structure
```
dawn-patrol/
├── .env             # stream url, Pushcut API keys, Pushcut URL, polygon mask path
├── main.py          # orchestration / count averaging / scheduler
├── capture.py       # HLS frame capture
├── detect.py        # masking, polygon filtering, YOLO inferencing 
├── notify.py        # Pushcut notifications
└── store.py         # logging counts to JSON/SQLite
```

## What I've Learned So Far
- Hosted video formats (HLS vs MJPEG)
- Practical CV tooling (OpenCV, Ultralytics, labelme)
- Why naive approaches fail on real-world scenes
- How object detection models work at a conceptual level
- Tradeoffs between custom model training and using a mask to reduce noise and improve detection
- Parsing object detection insights to push notification
- Making HTTP requests with Requests python library
- Using Pushcut to send iOS notifications

## Future Improvements
- Wave analysis using inference.
- Custom model training for more accurate surfer detection.
- Attach image capture to Pushcut notification.

## License
[MIT](LICENSE.md)
