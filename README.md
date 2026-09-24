# GestureOS
If computers can see what we're doing, do we always need to touch them to interact with them? 

GestureOS is a computer vision based project built with OpenCV and MediaPipe. It tracks your hand in real time and turns simple gestures into system actions like volume, brightness, and mouse control.

## Features

- 🎚️ **Volume Control** — pinch your thumb and index finger to adjust system volume
- ☀️ **Brightness Control** — same pinch gesture, applied to screen brightness
- 🖱️ **Air Mouse** — move your index finger to move the cursor, use finger counts to click
*(more currently in development)*

## How It Works

GestureOS uses MediaPipe's Hand Landmarker to detect 21 hand landmarks per frame, then maps their positions and distances to system actions. Modes are switched on the fly with number keys.

| Mode | Key | Action |
|------|-----|--------|
| 1 | `1` | Volume control |
| 2 | `2` | Brightness control |
| 3 | `3` | Air mouse |

**Air Mouse clicks:**
- ☝️ 1 finger up → left click
- 🤟 3 fingers up → right click

## Libraries used

- Python
- OpenCV
- MediaPipe (Hand Landmarker)
- PyAutoGUI
- Pycaw (system audio control)
- screen-brightness-control

## Getting Started

### Requirements

- Python 3.9+
- A webcam
- Windows *(required for `pycaw` and `screen_brightness_control`)*

### Installation

```bash
pip install opencv-python mediapipe pycaw screen-brightness-control keyboard pyautogui
```

You'll also need the `hand_landmarker.task` model file placed in the same directory as the script — download it from [MediaPipe's model page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

### Run

```bash
python gestureos.py
```

Press `q` to quit.

## Future Work

- User-trainable custom gestures
- More gesture-based features
- Cross-platform support
- Security features
- A modular SDK for developers

## Made by

Muhammad Ali - for the Security and Innovation Fair (SAIF)
