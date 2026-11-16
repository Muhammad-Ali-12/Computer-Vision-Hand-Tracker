# GestureOS

> **If computers can see what we're doing, do we always need to touch them to interact with them?**

GestureOS is a computer vision-based touchless human-computer interface built using Python, OpenCV, and MediaPipe. It tracks hand movements and gestures in real time, translating them into system-level actions such as volume adjustment, brightness control, cursor movement, and presentation navigation.

Designed with accessibility, convenience, and intuitive interaction in mind, GestureOS explores an alternative to traditional physical input devices through a unified gesture-controlled interface.

## Features

- 🎚️ **Volume Control** — Adjust system volume by spreading or pinching your thumb and index finger.
- ☀️ **Brightness Control** — Dynamically control screen brightness using the same intuitive pinch gesture.
- 🖱️ **Air Mouse** — Move the cursor using hand movements, with gesture-based left and right clicking.
- 🔒 **Gesture Lock Authentication** — Enroll a custom hand pose and use it to authenticate access to GestureOS.
- 👆 **Presentation Control** — Navigate slides using hand-tilt gestures, enabling touchless presentation control.
- 🔐 **Automatic Security Lock** — GestureOS automatically locks after 30 seconds without a detected hand.
- ✋ **Manual Gesture Lock** — Lock the interface by holding a designated hand gesture.
- 🔄 **Touchless Mode Switching** — Switch between modes using a thumbs-up gesture, without relying on physical input.
- 🖐️ **Real-Time Hand Tracking** — Detect and track 21 hand landmarks using MediaPipe's Hand Landmarker.

## How It Works

GestureOS uses MediaPipe's Hand Landmarker to detect 21 landmarks on the human hand in real time through a webcam.

These landmarks are processed using Python and OpenCV to recognize gestures and calculate hand movements. Based on the active mode, the detected gestures are translated into system actions.

### Gesture Controls

| Gesture | Action |
|---|---|
| Thumb + index finger spread/pinch | Adjust volume or brightness |
| Index + middle fingers raised | Move Air Mouse cursor |
| 1 finger raised | Left click |
| 3 fingers raised | Right click |
| Thumbs-up, then release | Switch to next mode |
| Enrolled hand pose | Unlock GestureOS |
| Hold designated pinky gesture for 2 seconds | Manually lock GestureOS |
| Hold thumbs-up for approximately 2 seconds | Re-enroll Gesture Lock pose |

### Gesture Lock

Gesture Lock is a custom authentication feature that allows users to enroll a hand pose of their choice.

The system captures and normalizes hand landmarks during enrollment, then compares subsequent poses against the saved template to determine whether access should be granted.

GestureOS also includes automatic relocking after 30 seconds without a detected hand, adding an additional layer of access control.

## Technologies & Libraries

- **Python** — Core programming language
- **OpenCV** — Webcam capture and visual interface
- **MediaPipe Hand Landmarker** — Real-time hand tracking and landmark detection
- **NumPy** — Landmark processing and pose comparison
- **PyAutoGUI** — System cursor and keyboard interaction
- **Pycaw** — Windows system audio control
- **Screen Brightness Control** — Display brightness adjustment

## Getting Started

### Requirements

- Python 3.9+
- Windows operating system
- Webcam
- Compatible display with brightness control support
- MediaPipe Hand Landmarker model

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/GestureOS.git
cd GestureOS
```

Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy pycaw screen-brightness-control pyautogui
```

Download the `hand_landmarker.task` model from the official MediaPipe model page:

[MediaPipe Hand Landmarker Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

Place the downloaded model file in the same directory as the Python script.

### Running GestureOS

```bash
python GestureOS.py
```

On the first launch, GestureOS will prompt you to enroll a custom hand pose for authentication.

Press `Q` in the camera window to exit the application.

## Future Work

- User-trainable and customizable gesture recognition
- Expanded gesture-based system controls
- Improved gesture recognition accuracy under varying environmental conditions
- Cross-platform compatibility
- Enhanced authentication and security mechanisms
- Modular SDK for third-party developers
- Integration with external hardware and IoT systems

## Project

Developed by **Muhammad Ali**

For the **Security and Innovation Fair (SAIF) 2026**, part of the Global Security and Technology Summit (GSTS).

GestureOS explores the potential of computer vision to make human-computer interaction more intuitive, accessible, and independent of traditional input devices.
