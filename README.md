# Dawn Patrol - In Progress

## Demo
Coming soon...

## Description
Daily text messages on surf conditions at my favorite local surf spot. Uses OpenCV to capture images from public beach camera footage, and YOLO object detection for image analysis.

## Context
I wanted to get my hands dirty with some of the open source image capture and object detection models/libraries. My local surf spot can be incredibly crowded, making the 45 minutes of traffic unappealing. This project gives me daily updates on surfer count so I don't have to check local feeds.

## Tech Stack
- Frontend: React, Vite, CSS
- Backend: Python, Flask, SQLAlchemy
- Database: SQLite
- Deployment: Fly.io, Docker
    - Deployed twice: once for myself, once for my partner

## Features
- Add, edit and delete transactions from a ledger.
- Live-updating summary including editable budget.
- Persistent data storage.

## Project Structure
```
dawn-patrol/
├── frames/          # storage for video frames
├── main.py          # entry point / scheduler
├── capture.py       # webcam frame grabbing
├── detect.py        # YOLO inference
├── notify.py        # SMS logic
└── store.py         # logging counts to JSON/SQLite
```

## What I've Learned So Far
- Hosted video formats (HLS vs MJPEG)
- OpenCV for frame capture from video 

## Future Improvements
Wave analysis using inference.

## License
[MIT](LICENSE)
