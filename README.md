git status


# AirSketch

AirSketch is a computer-vision prototype for drawing and manipulating geometric shapes in the air using hand gestures captured through a webcam.

## Features

- Real-time webcam hand tracking with MediaPipe
- Pinch-to-draw interaction
- Smoothed gesture tracking and pinch hysteresis
- Automatic recognition and snapping of:
  - Lines
  - Circles
  - Rectangles
  - Triangles
- Point-and-hold object selection
- Gesture-based object scaling
- Interaction states to reduce gesture conflicts
- Anti-aliased shape rendering
- Temporary hand-tracking loss protection

## Controls

| Action          | Gesture / Control                          |
| --------------- | ------------------------------------------ |
| Draw            | Pinch thumb + index finger and move        |
| Finish shape    | Release pinch                              |
| Select object   | Point at an object and hold                |
| Deselect        | Point into empty space and hold            |
| Prepare scaling | Hold an open palm near the selected object |
| Scale up        | Spread fingers                             |
| Scale down      | Bring fingers closer                       |
| Clear canvas    | `C`                                      |
| Quit            | `Q`                                      |

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks Vision
- NumPy

## Setup

```powershell
git clone https://github.com/shreysoni-analytics/AirSketch.git
cd AirSketch

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python Main.py
```
