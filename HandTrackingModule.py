import cv2
import mediapipe as mp

import math
import os

from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import keyboard
import pyautogui 

screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0

device = AudioUtilities.GetSpeakers()
Volume = device.EndpointVolume

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

mode = 1

frame_timestamp = 0
smooth_volume = 0
previous_fingers = 0

smooth_mouse_x = 0
smooth_mouse_y = 0

prev = set()

while True:
    keys = {k for k in ["1", "2", "3"] if keyboard.is_pressed(k)}
    
    if "1" in keys and "1" not in prev:
        mode = 1
    elif "2" in keys and "2" not in prev:
        mode = 2
    elif "3" in keys and "3" not in prev:
        mode = 3
    
    prev = keys

    success, img = cap.read()

    if not success:
        print("Failed to read camera")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )

    frame_timestamp += 1

    results = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:

            for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                h, w, c = img.shape

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            thumb_x, thumb_y = 0, 0
            index_x, index_y = 0, 0

            for id, lm in enumerate(hand_landmarks):
                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    (0, 0, 200),
                    cv2.FILLED
                )

                fingers_up = 0

                if hand_landmarks[8].y < hand_landmarks[6].y:
                    fingers_up += 1

                if hand_landmarks[12].y < hand_landmarks[10].y:
                    fingers_up += 1

                if hand_landmarks[16].y < hand_landmarks[14].y:
                    fingers_up += 1

                if hand_landmarks[20].y < hand_landmarks[18].y:
                    fingers_up += 1

                if id == 4:
                    thumb_x, thumb_y = cx, cy
                elif id == 8:
                    index_x, index_y = cx, cy

                    x = max(0, min(1, (lm.x - 0.2) / (0.8 - 0.2)))
                    y = max(0, min(1, (lm.y - 0.2) / (0.8 - 0.2)))

                    mouse_x = int((1 - x) * screen_width)
                    mouse_y = int(y * screen_height)

                    smooth_mouse_x = (smooth_mouse_x * (1 - 0.1)+ mouse_x * 0.25)
                    smooth_mouse_y = (smooth_mouse_y * (1 - 0.1) + mouse_y * 0.25)

                    smooth_mouse_x = max(10, min(screen_width - 15, smooth_mouse_x))
                    smooth_mouse_y = max(10, min(screen_height - 15, smooth_mouse_y))

            if mode == 1 or mode == 2:
                cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img,(thumb_x, thumb_y),(index_x, index_y),(255, 0, 255),3)
                
                distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                value = (distance - 25) / (200 - 25) * 100
                value = max(0, min(100, value))
    
                if value >= 100:
                    smooth_volume = 100
                elif value <= 0:
                    smooth_volume = 0

                smooth_volume = (smooth_volume * (1 - 0.3) + value * 0.3)

                if mode == 1: 
                    Volume.SetMasterVolumeLevelScalar(value / 100, None)
                    cv2.putText(
                        img,
                        f"Volume: {int(smooth_volume)}%",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        img,
                        f"Mode: Volume Control",
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                else: 
                    sbc.set_brightness(value)
                    cv2.putText(
                        img,
                        f"Brightness: {int(smooth_volume)}%",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        img,
                        f"Mode: Brightness Control",
                        (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
            else:
                cv2.circle(img, (index_x, index_y), 15, (255, 0, 255), cv2.FILLED)

                pyautogui.moveTo(
                    int(smooth_mouse_x),
                    int(smooth_mouse_y)
                )

                if fingers_up != previous_fingers:
                    if fingers_up == 1:
                        print("LEFT CLICK")
                        pyautogui.leftClick()

                    elif fingers_up == 3:
                        print("RIGHT CLICK")
                        pyautogui.rightClick()

                previous_fingers = fingers_up

                cv2.rectangle(
                    img,
                    (int(0.2 * w), int(0.2 * h)),
                    (int(0.8 * w), int(0.8 * h)),
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    img,
                    f"Mode: Air Mouse",
                    (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    img,
                    f"Fingers up: {int(fingers_up)}",
                    (350, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

    cv2.imshow("GestureOS", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()
