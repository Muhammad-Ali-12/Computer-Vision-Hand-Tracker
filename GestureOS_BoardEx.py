# ============================================================
# GESTURE OS - By Muhammad Ali
# SAIF 2026
# ============================================================

# Windows build: python -m PyInstaller --clean --noconfirm --onefile --console --name GestureOS_BoardEx --add-data "hand_landmarker.task;." --collect-all mediapipe --hidden-import mediapipe.tasks.c GestureOS_BoardEx.py
import cv2
import mediapipe as mp
import math
import sys
import time
import json
import numpy as np
import pyautogui
from pycaw.pycaw import AudioUtilities
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CAMERA_INDICES = [0, 1, 2, 3]
MODE_NAMES = {
    1: "Volume Control",
    2: "Presentation Control",
}

MODE_HELP = {
    1: [
        "Spread thumb and index finger to raise volume; pinch them closer to lower it.",
        "Thumbs-up, then lower your hand to switch to the next mode.",
    ],
    2: [
        "Tilt right = next/down; tilt left = previous/up (one action per tilt).",
        "Return upright OR lower your hand before tilting again. Thumbs-up = next mode.",
    ],
}

# In a PyInstaller one-file build, read bundled assets from _MEIPASS but save
# the enrolled template beside the EXE so it persists across launches.
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
    RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    APP_DIR = Path(__file__).resolve().parent
    RESOURCE_DIR = APP_DIR

TEMPLATE_PATH = APP_DIR / "gesture_lock_template.json"
MODEL_PATH = str(RESOURCE_DIR / "hand_landmarker.task")

LOCK_TIMEOUT_SECONDS = 30
LOCK_GESTURE_HOLD_SECONDS = 2.0
LOCK_GESTURE_LOSS_GRACE_SECONDS = 0.18
LOCK_GESTURE_RELEASE_SECONDS = 0.35
ENROLL_COUNTDOWN_SECONDS = 5
ENROLL_STABLE_FRAMES = 20
ENROLL_STABILITY_THRESHOLD = 0.12
AUTH_MATCH_THRESHOLD = 0.22
AUTH_REQUIRED_FRAMES = 5
THUMBS_UP_MIN_HOLD = 0.35
THUMBS_UP_RELEASE_GRACE = 0.18
REENROLL_HOLD_SECONDS = 1.8

# Generic navigation works with many slideshow apps and PDF readers.
# "page" = Page Down/Page Up (default); "arrows" = Right/Left;
# "wheel" = mouse-wheel down/up (use only for readers that need it).
PRESENTATION_OUTPUT = "page"
PRESENTATION_SCROLL_STEPS = 4
PRESENTATION_COOLDOWN_SECONDS = 0.30
PRESENTATION_NEUTRAL_ANGLE = 10
PRESENTATION_RIGHT_THRESHOLD = 19
PRESENTATION_LEFT_THRESHOLD = -30
PRESENTATION_TILT_HOLD_SECONDS = 0.12
PRESENTATION_NEUTRAL_HOLD_SECONDS = 0.16
PRESENTATION_HAND_AWAY_SECONDS = 0.28
PRESENTATION_SMOOTHING = 0.55

mode = 1
frame_timestamp = 0
pyautogui.PAUSE = 0

security_unlocked = False
template = None
security_status = "LOCKED - no pose enrolled; thumbs-up to start enrollment"
auth_streak = 0
last_pose_distance = None
last_hand_seen_time = time.monotonic()

enroll_phase = None  # None, "countdown", or "capture"
enrollment_started_at = 0.0
enrollment_samples = []
enrollment_reference_pose = None

thumbsup_start_time = None
thumbsup_last_seen_time = 0.0

lock_gesture_started_at = None
lock_gesture_last_seen_at = 0.0
lock_release_started_at = None
lock_waiting_for_release = False

smooth_value = 0.0

presentation_last_trigger_time = 0.0
presentation_armed = True
presentation_action = "READY"
presentation_smooth_angle = None
presentation_direction = None
presentation_direction_since = None
presentation_neutral_since = None
presentation_hand_away_since = None

# ============================================================
# Gesture and pose helpers
# ============================================================

def normalized_pose(world_landmarks):
    """Normalize a 21-point world-landmark pose for position/scale comparison."""
    points = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float32)
    if points.shape != (21, 3):
        return None
    points -= points[0]  # Wrist becomes the origin.
    scale = float(np.linalg.norm(points[9]))  # Wrist to middle-finger MCP.
    if not np.isfinite(scale) or scale < 1e-5:
        return None
    points /= scale
    if not np.all(np.isfinite(points)):
        return None
    return points.reshape(-1)


def load_pose_template():
    if not TEMPLATE_PATH.exists():
        return None
    try:
        data = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        pose = np.asarray(data["pose"], dtype=np.float32)
        if pose.shape != (63,) or not np.all(np.isfinite(pose)):
            raise ValueError("Expected 63 finite landmark values")
        return pose
    except Exception as exc:
        print(f"Could not load gesture-lock template: {exc}")
        return None


def save_pose_template(pose):
    payload = {
        "format": 1,
        "description": "Normalized MediaPipe hand-pose template",
        "pose": [float(value) for value in pose],
    }
    TEMPLATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_pose(current, saved):
    current_points = current.reshape(21, 3)
    saved_points = saved.reshape(21, 3)
    per_landmark = np.linalg.norm(current_points - saved_points, axis=1)
    return float(np.sqrt(np.mean(per_landmark ** 2)))


def is_thumbs_up(landmarks):
    """Recognize an upright thumb with the other four fingers curled."""
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    # The thumb should be extended and pointing upward in the camera image.
    thumb_tip_dist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    thumb_mcp_dist = math.hypot(thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y)
    thumb_extended = (
        thumb_tip.y < thumb_ip.y - 0.025
        and thumb_tip_dist > thumb_mcp_dist * 1.12
    )

    # For the four curled fingers, fingertip-to-wrist distance should not be
    # substantially greater than the corresponding PIP-joint distance.
    curled = True
    for tip_id, pip_id in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        tip_dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        pip_dist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        if tip_dist > pip_dist * 1.12:
            curled = False
            break

    return thumb_extended and curled


def is_lock_gesture(landmarks):
    """pinky extended locks the program"""
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]

    def finger_extended(tip_id, pip_id):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        tip_distance = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        pip_distance = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
        return tip_distance > pip_distance * 1.14

    return (
        not finger_extended(8, 6)         # Index folded
        and not finger_extended(12, 10)  # Middle folded
        and not finger_extended(16, 14)  # Ring folded
        and finger_extended(20, 18)   # Pinky raised
    )


def reset_presentation_tracking():
    """Reset a partial tilt when switching modes or locking the program."""
    global presentation_armed, presentation_action, presentation_smooth_angle
    global presentation_direction, presentation_direction_since
    global presentation_neutral_since, presentation_hand_away_since

    presentation_armed = True
    presentation_action = "READY"
    presentation_smooth_angle = None
    presentation_direction = None
    presentation_direction_since = None
    presentation_neutral_since = None
    presentation_hand_away_since = None


def send_presentation_command(direction):
    """Send exactly one generic navigation command to the active application."""
    if PRESENTATION_OUTPUT == "page":
        pyautogui.press("pagedown" if direction == "right" else "pageup")
    elif PRESENTATION_OUTPUT == "arrows":
        pyautogui.press("right" if direction == "right" else "left")
    elif PRESENTATION_OUTPUT == "wheel":
        pyautogui.scroll(-PRESENTATION_SCROLL_STEPS if direction == "right" else PRESENTATION_SCROLL_STEPS)
    else:
        raise ValueError("PRESENTATION_OUTPUT must be 'page', 'arrows', or 'wheel'")


def update_presentation(landmarks, now):
    """One command per stable tilt; rearm after neutral or deliberate hand absence.

    Timing and a light angle filter reject single-frame jitter. This deliberately
    uses the original wrist-to-middle-MCP angle, not finger-count recognition.
    """
    global presentation_armed, presentation_action, presentation_smooth_angle
    global presentation_direction, presentation_direction_since
    global presentation_neutral_since, presentation_hand_away_since
    global presentation_last_trigger_time

    if not landmarks:
        if presentation_hand_away_since is None:
            presentation_hand_away_since = now
        presentation_smooth_angle = None
        presentation_direction = None
        presentation_direction_since = None
        presentation_neutral_since = None
        if now - presentation_hand_away_since >= PRESENTATION_HAND_AWAY_SECONDS:
            presentation_armed = True
            presentation_action = "READY - HAND LOWERED"
        else:
            presentation_action = "HAND NOT VISIBLE"
        return None

    presentation_hand_away_since = None
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    # Preserve v1's direction and hand-tilt geometry (unmirrored camera).
    dx = wrist.x - middle_mcp.x
    dy = wrist.y - middle_mcp.y
    raw_angle = math.degrees(math.atan2(dx, dy))
    if presentation_smooth_angle is None:
        presentation_smooth_angle = raw_angle
    else:
        presentation_smooth_angle += PRESENTATION_SMOOTHING * (raw_angle - presentation_smooth_angle)
    angle = presentation_smooth_angle

    if abs(angle) <= PRESENTATION_NEUTRAL_ANGLE:
        presentation_direction = None
        presentation_direction_since = None
        if presentation_neutral_since is None:
            presentation_neutral_since = now
        if now - presentation_neutral_since >= PRESENTATION_NEUTRAL_HOLD_SECONDS:
            presentation_armed = True
            presentation_action = "READY"
        else:
            presentation_action = "HOLD UPRIGHT" if not presentation_armed else "READY"
        return angle

    presentation_neutral_since = None
    if not presentation_armed:
        presentation_direction = None
        presentation_direction_since = None
        presentation_action = "RETURN UPRIGHT OR LOWER HAND"
        return angle

    direction = ("right" if angle > PRESENTATION_RIGHT_THRESHOLD else
                 "left" if angle < PRESENTATION_LEFT_THRESHOLD else None)
    if direction is None:
        presentation_direction = None
        presentation_direction_since = None
        presentation_action = "TILT FURTHER"
        return angle

    if direction != presentation_direction:
        presentation_direction = direction
        presentation_direction_since = now
        presentation_action = "HOLD TILT BRIEFLY"
        return angle

    if now - presentation_direction_since < PRESENTATION_TILT_HOLD_SECONDS:
        presentation_action = "HOLD TILT BRIEFLY"
        return angle
    if now - presentation_last_trigger_time < PRESENTATION_COOLDOWN_SECONDS:
        presentation_action = "COOLDOWN"
        return angle

    try:
        send_presentation_command(direction)
    except Exception as exc:
        presentation_last_trigger_time = now  # Don't retry continuously on error.
        presentation_action = "NAVIGATION ERROR"
        print(f"Presentation command failed: {exc}")
    else:
        presentation_last_trigger_time = now
        presentation_armed = False
        presentation_action = "NEXT / DOWN" if direction == "right" else "PREVIOUS / UP"
        print(f"Presentation Control: {presentation_action}")
    return angle


def begin_enrollment(now, reason):
    global enroll_phase, enrollment_started_at, enrollment_samples
    global enrollment_reference_pose, security_unlocked, auth_streak
    global last_pose_distance, security_status

    enroll_phase = "countdown"
    enrollment_started_at = now
    enrollment_samples = []
    enrollment_reference_pose = None
    security_unlocked = False
    auth_streak = 0
    last_pose_distance = None
    security_status = "ENROLLMENT STARTING - show your chosen pose"
    print(f"Gesture Lock enrollment started ({reason}). Five-second countdown.")


def handle_thumbs_up(duration, now):
    """A short thumbs-up advances modes; a long one starts re-enrollment."""
    global mode, security_status

    if enroll_phase is not None:
        return

    if template is None:
        begin_enrollment(now, "first-time enrollment")
        return

    if not security_unlocked:
        security_status = "LOCKED - perform your saved pose to unlock"
        return

    if duration >= REENROLL_HOLD_SECONDS:
        begin_enrollment(now, "re-enrollment requested by long thumbs-up")
        return

    mode = 2 if mode == 1 else 1
    reset_presentation_tracking()
    print(f"Mode changed to {mode}: {MODE_NAMES[mode]}")


def initialize_camera():
    print("Searching for available camera...")
    for camera_index in CAMERA_INDICES:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Camera index {camera_index} could not be opened.")
            cap.release()
            continue
        success, frame = cap.read()
        if success and frame is not None:
            print(f"Camera index {camera_index} connected.")
            return cap
        print(f"Camera index {camera_index} opened but did not provide frames.")
        cap.release()
    return None


# ============================================================
# HUD/help panel
# ============================================================

def draw_text(img, text, x, y, scale=0.52, color=(255, 255, 255), thickness=1):
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def draw_hud(img, mode_number, status, enrollment_phase, enrollment_started):
    h, w = img.shape[:2]
    panel_height = min(132, max(100, h // 3))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - panel_height), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    title = f"GestureOS | MODE {mode_number}/2: {MODE_NAMES[mode_number]}"
    draw_text(img, title, 12, h - panel_height + 20, 0.56, (0, 255, 255), 2)

    if enrollment_phase == "countdown":
        remaining = max(0, int(math.ceil(ENROLL_COUNTDOWN_SECONDS - (time.monotonic() - enrollment_started))))
        help1 = f"GESTURE LOCK ENROLLMENT: perform your chosen pose now. Capture starts in {remaining}s."
        help2 = "After the countdown, keep your hand still until the pose is saved."
    elif enrollment_phase == "capture":
        help1 = "GESTURE LOCK ENROLLMENT: KEEP YOUR HAND STILL while GestureOS captures the pose."
        help2 = "If your hand moves, the sample counter resets. Hold one hand clearly in view."
    elif not security_unlocked:
        if template is None:
            help1 = "SECURITY: no saved pose. Make a thumbs-up, then lower your hand to start enrollment."
            help2 = "You will get 5 seconds to prepare your pose, then hold it still for capture."
        else:
            help1 = "SECURITY LOCKED: show the saved hand pose and hold it steady to unlock."
            help2 = "Mode changes are available after authentication."
    else:
        lines = MODE_HELP[mode_number]
        help1, help2 = lines[0], lines[1]
        if mode_number == 1 and not security_unlocked:
            help1 = "SECURITY LOCKED"

    draw_text(img, help1[:105], 12, h - panel_height + 48, 0.43, (255, 255, 255), 1)
    draw_text(img, help2[:105], 12, h - panel_height + 69, 0.43, (220, 220, 220), 1)

    if security_unlocked and enrollment_phase is None:
        draw_text(
            img,
            "Thumbs-up + release = next mode | Hold your pinky finger up for 2s to lock.",
            12, h - 10, 0.38, (180, 230, 180), 1,
        )
    else:
        draw_text(img, "GestureOS uses your saved pose for access. Press Q in the camera window to exit.",
                  12, h - 10, 0.38, (180, 230, 180), 1)

    status_color = (70, 230, 100) if security_unlocked else (0, 190, 255)
    draw_text(img, f"Security: {status}", 12, 24, 0.53, status_color, 2)

# ============================================================
# Startup
# ============================================================

print("=" * 54)
print("Starting GestureOS - updated touchless build")
print("Modes: 1 Volume | 2 Presentation Control")
print("Thumbs-up then lower hand: next mode; long thumbs-up (~2s) while unlocked: re-enroll lock")
print("Hold index + pinky up for 2 seconds to lock instantly. Lower your hand before unlocking.")
print("=" * 54)

cap = initialize_camera()
if cap is None:
    raise RuntimeError(
        "GestureOS could not access a webcam. Check the USB connection, "
        "camera permissions, and camera index."
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)
print("MediaPipe Hand Landmarker initialized.")

template = load_pose_template()
if template is None:
    security_status = "LOCKED - no pose enrolled; thumbs-up to start enrollment"
    print("No valid Gesture Lock template. Use thumbs-up to start enrollment.")
else:
    security_status = "LOCKED - perform your saved gesture"
    print("Saved Gesture Lock pose loaded. GestureOS starts locked.")

audio_device = AudioUtilities.GetSpeakers()
volume = audio_device.EndpointVolume


# ============================================================
# Main loop
# ============================================================

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Camera frame could not be read.")
            time.sleep(0.05)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        frame_timestamp = max(frame_timestamp + 1, time.monotonic_ns() // 1_000_000)
        results = detector.detect_for_video(mp_image, frame_timestamp)
        now = time.monotonic()

        has_hand = bool(results.hand_landmarks) and bool(results.hand_world_landmarks)
        landmarks = results.hand_landmarks[0] if results.hand_landmarks else None
        world_landmarks = results.hand_world_landmarks[0] if results.hand_world_landmarks else None
        current_pose = normalized_pose(world_landmarks) if world_landmarks else None
        thumbs_up_now = is_thumbs_up(landmarks) if landmarks else False
        lock_gesture_now = is_lock_gesture(landmarks) if landmarks else False

        # Detect a complete thumbs-up gesture (hold, then lower/release).
        # It is intentionally edge-triggered so one held pose changes mode once.
        if thumbs_up_now:
            if thumbsup_start_time is None:
                thumbsup_start_time = now
            thumbsup_last_seen_time = now
        elif thumbsup_start_time is not None:
            if now - thumbsup_last_seen_time >= THUMBS_UP_RELEASE_GRACE:
                duration = thumbsup_last_seen_time - thumbsup_start_time
                if duration >= THUMBS_UP_MIN_HOLD:
                    handle_thumbs_up(duration, now)
                thumbsup_start_time = None
                thumbsup_last_seen_time = 0.0

        # Authentication/enrollment gate.
        if has_hand:
            last_hand_seen_time = now

        if enroll_phase == "countdown":
            security_unlocked = False
            remaining = ENROLL_COUNTDOWN_SECONDS - (now - enrollment_started_at)
            security_status = f"ENROLLMENT - show chosen pose; capture in {max(0, int(math.ceil(remaining)))}s"
            if remaining <= 0:
                enroll_phase = "capture"
                enrollment_samples = []
                enrollment_reference_pose = None
                security_status = "ENROLLMENT - keep your hand still"
                print("Enrollment countdown finished. Capturing a stable hand pose.")

        elif enroll_phase == "capture":
            security_unlocked = False
            if current_pose is None:
                enrollment_samples = []
                enrollment_reference_pose = None
                security_status = "ENROLLMENT - show one hand clearly"
            else:
                if enrollment_reference_pose is None:
                    enrollment_reference_pose = current_pose.copy()
                    enrollment_samples = [current_pose.copy()]
                else:
                    distance = compare_pose(current_pose, enrollment_reference_pose)
                    if distance <= ENROLL_STABILITY_THRESHOLD:
                        enrollment_samples.append(current_pose.copy())
                        enrollment_reference_pose = np.median(
                            np.stack(enrollment_samples[-10:]), axis=0
                        ).astype(np.float32)
                    else:
                        enrollment_samples = [current_pose.copy()]
                        enrollment_reference_pose = current_pose.copy()

                security_status = f"ENROLLMENT - keep still: {len(enrollment_samples)}/{ENROLL_STABLE_FRAMES}"
                if len(enrollment_samples) >= ENROLL_STABLE_FRAMES:
                    new_template = np.median(
                        np.stack(enrollment_samples[:ENROLL_STABLE_FRAMES]), axis=0
                    ).astype(np.float32)
                    try:
                        save_pose_template(new_template)
                        template = new_template
                        enroll_phase = None
                        enrollment_samples = []
                        enrollment_reference_pose = None
                        auth_streak = 0
                        last_pose_distance = None
                        security_unlocked = False
                        security_status = "POSE SAVED - perform it to unlock"
                        print(f"Gesture Lock pose saved to: {TEMPLATE_PATH}")
                    except Exception as exc:
                        security_status = f"ENROLLMENT SAVE ERROR: {exc}"
                        print(f"Could not save Gesture Lock pose: {exc}")

        elif not security_unlocked:
            if lock_waiting_for_release:
                # Require the hand to be lowered after a manual lock. This
                # prevents the held lock gesture from unlocking again if
                # someone happened to enroll the same pose.
                if not has_hand:
                    if lock_release_started_at is None:
                        lock_release_started_at = now
                    elif now - lock_release_started_at >= LOCK_GESTURE_RELEASE_SECONDS:
                        lock_waiting_for_release = False
                        lock_release_started_at = None
                else:
                    lock_release_started_at = None
                auth_streak = 0
                last_pose_distance = None
                security_status = (
                    "LOCKED - perform your saved pose to unlock"
                    if not lock_waiting_for_release
                    else "LOCKED - lower your hand, then authenticate"
                )
            elif template is None:
                security_status = "LOCKED - thumbs-up to enroll a pose"
                auth_streak = 0
                last_pose_distance = None
            elif current_pose is None:
                security_status = "LOCKED - show your saved gesture"
                auth_streak = 0
                last_pose_distance = None
            else:
                last_pose_distance = compare_pose(current_pose, template)
                if last_pose_distance <= AUTH_MATCH_THRESHOLD:
                    auth_streak += 1
                    if auth_streak >= AUTH_REQUIRED_FRAMES:
                        security_unlocked = True
                        last_hand_seen_time = now
                        security_status = "UNLOCKED - gesture authenticated"
                        print("GestureOS unlocked: gesture authenticated.")
                    else:
                        security_status = f"VERIFYING - {auth_streak}/{AUTH_REQUIRED_FRAMES}"
                else:
                    auth_streak = 0
                    security_status = "LOCKED - gesture not recognized"

        else:
            if has_hand:
                last_hand_seen_time = now
                security_status = "UNLOCKED"
            elif now - last_hand_seen_time >= LOCK_TIMEOUT_SECONDS:
                security_unlocked = False
                auth_streak = 0
                last_pose_distance = None
                security_status = "LOCKED - hand absent; authenticate again"
                print(f"GestureOS relocked after {LOCK_TIMEOUT_SECONDS} seconds without a detected hand.")
            else:
                remaining = max(0, int(math.ceil(LOCK_TIMEOUT_SECONDS - (now - last_hand_seen_time))))
                security_status = f"UNLOCKED - hand timeout: {remaining}s"

        # Manual touchless lock: hold the distinct index+pinky gesture for 2s.
        # It only runs when already authenticated and not enrolling.
        if security_unlocked and enroll_phase is None and lock_gesture_now:
            if lock_gesture_started_at is None:
                lock_gesture_started_at = now
            lock_gesture_last_seen_at = now
            elapsed = now - lock_gesture_started_at
            if elapsed >= LOCK_GESTURE_HOLD_SECONDS:
                security_unlocked = False
                auth_streak = 0
                last_pose_distance = None
                lock_gesture_started_at = None
                lock_waiting_for_release = True
                lock_release_started_at = None
                security_status = "LOCKED - lower your hand, then authenticate"
                print("GestureOS locked manually (index + pinky gesture).")
            else:
                remaining = LOCK_GESTURE_HOLD_SECONDS - elapsed
                security_status = f"LOCK gesture - hold {remaining:.1f}s more"
        elif (
            lock_gesture_started_at is not None
            and security_unlocked
            and now - lock_gesture_last_seen_at <= LOCK_GESTURE_LOSS_GRACE_SECONDS
        ):
            # Ignore a brief single-frame recognition dropout.
            pass
        else:
            lock_gesture_started_at = None

        # Draw the detected hand skeleton. All modes use one primary hand.
        frame_h, frame_w = img.shape[:2]
        if landmarks:
            for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                start = landmarks[connection.start]
                end = landmarks[connection.end]
                cv2.line(
                    img,
                    (int(start.x * frame_w), int(start.y * frame_h)),
                    (int(end.x * frame_w), int(end.y * frame_h)),
                    (0, 255, 0), 2,
                )
            for lm in landmarks:
                cv2.circle(img, (int(lm.x * frame_w), int(lm.y * frame_h)), 4, (0, 0, 220), -1)

        # A brief lost frame must NOT re-arm a finished tilt. A deliberately
        # lowered hand can re-arm it after PRESENTATION_HAND_AWAY_SECONDS.
        if mode != 2 or not security_unlocked or enroll_phase is not None:
            reset_presentation_tracking()
            if mode == 2 and not security_unlocked:
                presentation_action = "LOCKED - authenticate first"
        elif not landmarks:
            update_presentation(None, now)
        elif thumbs_up_now or lock_gesture_started_at is not None:
            # Reserved gestures must never also advance a slide.
            reset_presentation_tracking()

        # Execute mode actions only after Gesture Lock authentication. A thumbs-up
        # is reserved for navigation, so it will not accidentally adjust volume.
        if security_unlocked and landmarks and enroll_phase is None and lock_gesture_started_at is None:
            if thumbs_up_now:
                pass
            elif mode == 1:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                tx = int(thumb_tip.x * frame_w)
                ty = int(thumb_tip.y * frame_h)
                ix = int(index_tip.x * frame_w)
                iy = int(index_tip.y * frame_h)
                cv2.circle(img, (tx, ty), 12, (255, 0, 255), -1)
                cv2.circle(img, (ix, iy), 12, (255, 0, 255), -1)
                cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 3)

                # Normalize the pinch by palm width so moving nearer/farther
                # from the camera does not change the volume by itself.
                palm_width = math.hypot(
                    landmarks[5].x - landmarks[17].x,
                    landmarks[5].y - landmarks[17].y,
                )
                if palm_width > 1e-5:
                    pinch = math.hypot(
                        thumb_tip.x - index_tip.x,
                        thumb_tip.y - index_tip.y,
                    ) / palm_width
                    value = max(0.0, min(100.0, (pinch - 0.20) / (1.65 - 0.20) * 100.0))
                    # Snap endpoints exactly, avoiding 99% at maximum spread.
                    if value >= 98.0:
                        smooth_value = 100.0
                    elif value <= 2.0:
                        smooth_value = 0.0
                    else:
                        smooth_value = smooth_value * 0.7 + value * 0.3
                    try:
                        volume.SetMasterVolumeLevelScalar(smooth_value / 100.0, None)
                    except Exception as exc:
                        security_status = f"Volume control error: {exc}"
                draw_text(img, f"Volume: {round(smooth_value)}%", 12, 54, 0.65, (255, 255, 255), 2)

            elif mode == 2:
                presentation_angle = update_presentation(landmarks, now)
                draw_text(img, f"Tilt: {presentation_angle:.1f} deg | {presentation_action}",
                          12, 54, 0.56, (255, 255, 255), 2)


        draw_hud(img, mode, security_status, enroll_phase, enrollment_started_at)
        cv2.imshow("GestureOS", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    print("Shutting down GestureOS...")
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("GestureOS closed.")
