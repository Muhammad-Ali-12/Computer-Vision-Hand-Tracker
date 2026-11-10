"""import cv2
import os
import time
import math
import serial
import mediapipe as mp

ARDUINO_PORT = "COM7"
BAUD_RATE = 9600

arduino = serial.Serial(
    ARDUINO_PORT,
    BAUD_RATE
)

time.sleep(2)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print("Could not open camera.")

    arduino.close()

    exit()


last_finger_count = None

timestamp_ms = 0

# ============================================================
# MAIN
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    try:

        while True:

            success, frame = cap.read()

            if not success:

                print("Could not read camera frame.")

                break


            # ------------------------------------------------
            # MIRROR CAMERA
            # ------------------------------------------------

            frame = cv2.flip(frame, 1)


            # ------------------------------------------------
            # CONVERT TO MEDIAPIPE IMAGE
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp_ms += 33


            # ------------------------------------------------
            # DETECT HAND
            # ------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            finger_count = 0

            finger_states = []

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]


                # ---------------------------------------------
                # GET FINGER STATES
                # ---------------------------------------------

                #finger_states = get_finger_states(hand)
                thumb_tip = hand[4]
                thumb_ip = hand[3]
            
                index_tip = hand[8]
                index_pip = hand[6]
            
                middle_tip = hand[12]
                middle_pip = hand[10]
            
                ring_tip = hand[16]
                ring_pip = hand[14]
            
                pinky_tip = hand[20]
                pinky_pip = hand[18]
            
                index_up = index_tip.y < index_pip.y
            
                middle_up = middle_tip.y < middle_pip.y
            
                ring_up = ring_tip.y < ring_pip.y
            
                pinky_up = pinky_tip.y < pinky_pip.y
            
                index_mcp = hand[5]
            
                thumb_distance = math.hypot(
                    thumb_tip.x - index_mcp.x,
                    thumb_tip.y - index_mcp.y
                )
            
                thumb_up = thumb_distance > 0.12
            
            
                finger_states = [
                    thumb_up,
                    index_up,
                    middle_up,
                    ring_up,
                    pinky_up
                ]

                finger_count = sum(
                    finger_states
                )

                h, w, _ = frame.shape

                for landmark in hand:

                    x = int(landmark.x * w)

                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            # Only send when the finger count changes.
            # This prevents sending data every frame.

            if finger_count != last_finger_count:

                command = str(finger_count)

                arduino.write(
                    command.encode()
                )

                last_finger_count = finger_count

                print(
                    f"Fingers: {finger_count} | "
                    f"Arduino command: {command}"
                )

            cv2.putText(
                frame,
                f"Fingers: {finger_count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            if finger_states:

                status = (
                    f"T:{int(finger_states[0])} "
                    f"I:{int(finger_states[1])} "
                    f"M:{int(finger_states[2])} "
                    f"R:{int(finger_states[3])} "
                    f"P:{int(finger_states[4])}"
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                f"LEDs ON: {finger_count}/5",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "GestureOS - Arduino Finger Tracker",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):

                break


    finally:

        try:
            arduino.write(b"0")

        except:
            pass


        cap.release()

        cv2.destroyAllWindows()

        arduino.close()"""

import cv2
import os
import time
import math
import serial
import mediapipe as mp

# ============================================================
# ARDUINO
# ============================================================

ARDUINO_PORT = "COM7"
BAUD_RATE = 9600

arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)

time.sleep(2)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open camera.")
    arduino.close()
    exit()


last_state = None
timestamp_ms = 0

def fingers_up(hand):

    # MediaPipe landmark indexes:
    #
    # Thumb:
    # 4 = tip
    # 2 = IP joint
    #
    # Index:
    # 8 = tip
    # 6 = PIP
    #
    # Middle:
    # 12 = tip
    # 10 = PIP
    #
    # Ring:
    # 16 = tip
    # 14 = PIP
    #
    # Pinky:
    # 20 = tip
    # 18 = PIP

    thumb_tip = hand[4]
    thumb_ip = hand[3]

    index_tip = hand[8]
    index_pip = hand[6]

    middle_tip = hand[12]
    middle_pip = hand[10]

    ring_tip = hand[16]
    ring_pip = hand[14]

    pinky_tip = hand[20]
    pinky_pip = hand[18]

    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    index_mcp = hand[5]

    thumb_distance = math.hypot(
        thumb_tip.x - index_mcp.x,
        thumb_tip.y - index_mcp.y
    )

    thumb_up = thumb_distance > 0.12

    return (
        thumb_up,
        index_up,
        middle_up,
        ring_up,
        pinky_up
    )

with HandLandmarker.create_from_options(options) as landmarker:

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Could not read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            all_fingers_up = False

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                (
                    thumb,
                    index,
                    middle,
                    ring,
                    pinky
                ) = fingers_up(hand)

                all_fingers_up = (
                    thumb and
                    index and
                    middle and
                    ring and
                    pinky
                )

                h, w, _ = frame.shape

                for landmark in hand:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                finger_text = (
                    f"T:{int(thumb)} "
                    f"I:{int(index)} "
                    f"M:{int(middle)} "
                    f"R:{int(ring)} "
                    f"P:{int(pinky)}"
                )

                cv2.putText(
                    frame,
                    finger_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if all_fingers_up:
                state = "1"
            else:
                state = "0"

            if state != last_state:

                arduino.write(state.encode())

                last_state = state

                print(
                    "LED ON"
                    if state == "1"
                    else "LED OFF"
                )

            if all_fingers_up:

                status = "LED ON - ALL FINGERS UP"

                status_color = (0, 255, 0)

            else:

                status = "LED OFF"

                status_color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            cv2.imshow(
                "GestureOS - Arduino Control",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            arduino.write(b"0")
        except:
            pass

        cap.release()
        cv2.destroyAllWindows()
        arduino.close()