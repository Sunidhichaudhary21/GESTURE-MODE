import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui
import queue
import re
import subprocess
import threading
import time
import urllib.request
import ctypes
from datetime import datetime
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pyautogui.FAILSAFE = False

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model
except ImportError:
    sd = None
    KaldiRecognizer = None
    Model = None

# ---------------------------
# Volume setup
# ---------------------------
devices = AudioUtilities.GetSpeakers()

# Support both legacy and newer pycaw device wrappers.
if hasattr(devices, "Activate"):
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
elif hasattr(devices, "EndpointVolume"):
    volume = devices.EndpointVolume
else:
    raise RuntimeError("Could not access audio endpoint volume via pycaw")

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# ---------------------------
# Mediapipe setup
# ---------------------------
try:
    from mediapipe.tasks import python as mp_python_tasks
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import mediapipe_c_bindings as mp_c_bindings
except ImportError as exc:
    raise RuntimeError(
        "Installed mediapipe package does not include the Tasks API. "
        "Install dependencies from requirement.txt (mediapipe==0.10.30)."
    ) from exc


def patch_mediapipe_free_symbol():
    """Work around missing free() export in some Windows MediaPipe builds."""

    original_load_raw_library = mp_c_bindings.load_raw_library

    def load_raw_library_with_free_fallback(signatures=()):
        try:
            return original_load_raw_library(signatures)
        except AttributeError as exc:
            if "function 'free' not found" not in str(exc):
                raise

            shared_lib = mp_c_bindings._shared_lib
            if shared_lib is None:
                raise

            for runtime_lib in ("ucrtbase", "msvcrt"):
                try:
                    crt = ctypes.CDLL(runtime_lib)
                    crt_free = crt.free
                    crt_free.argtypes = [ctypes.c_void_p]
                    crt_free.restype = None
                    shared_lib.free = crt_free
                    return shared_lib
                except (AttributeError, OSError):
                    continue

            def noop_free(_ptr):
                return None

            shared_lib.free = noop_free
            return shared_lib

    mp_c_bindings.load_raw_library = load_raw_library_with_free_fallback


patch_mediapipe_free_symbol()

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

HAND_MODEL_PATH = os.getenv("HAND_MODEL_PATH", r"S:\New folder\hand_landmarker.task")
HAND_MODEL_URL = os.getenv(
    "HAND_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
)
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", r"S:\New folder\face_landmarker.task")
FACE_MODEL_URL = os.getenv(
    "FACE_MODEL_URL",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
)


def ensure_hand_model(model_path):
    if os.path.isfile(model_path):
        return

    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download hand model from {HAND_MODEL_URL}. "
            "Set HAND_MODEL_PATH to a local hand_landmarker.task file."
        ) from exc


ensure_hand_model(HAND_MODEL_PATH)


def ensure_face_model(model_path):
    if os.path.isfile(model_path):
        return

    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to download face model from {FACE_MODEL_URL}. "
            "Set FACE_MODEL_PATH to a local face_landmarker.task file."
        ) from exc


ensure_face_model(FACE_MODEL_PATH)

base_options = mp_python_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
    running_mode=mp_vision.RunningMode.IMAGE,
)

hand_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python_tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=mp_vision.RunningMode.IMAGE,
)
face_detector = mp_vision.FaceLandmarker.create_from_options(face_options)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_LIDS = (159, 145)
RIGHT_EYE_LIDS = (386, 374)

# ---------------------------
# Camera and app state
# ---------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)
last_mute_toggle_time = 0.0
last_screenshot_time = 0.0
last_lock_toggle_time = 0.0
mute_cooldown_sec = 1.0
screenshot_cooldown_sec = 2.0
lock_cooldown_sec = 1.5
fist_hold_sec = 1.0

controls_locked = False
fist_hold_start = None
smoothed_vol = None
volume_alpha = 0.25
last_frame_time = time.time()

screen_w, screen_h = pyautogui.size()
gaze_smooth = None
last_gaze_point = None
gaze_alpha_fast = 0.4
gaze_alpha_slow = 0.05
gaze_fast_threshold = 35  # Pixels

# Calibration state
CALIBRATION_POINTS = []
CALIBRATION_DATA = []
is_calibrating = False
calibration_step = 0
calibration_margin = 50
calibration_finished = False

blink_ratio_smooth = None
blink_alpha = 0.75
blink_click_threshold = 0.25
last_blink_click_time = 0.0
blink_click_cooldown_sec = 0.5
double_blink_time_sec = 0.4  # Time window for a double blink
last_blink_time = 0.0
blink_state = "NO_BLINK"  # NO_BLINK, BLINK_1, BLINK_2

CONTROL_MODE_ALIASES = {
    "1": "hand",
    "2": "eye",
    "3": "both",
    "hand": "hand",
    "eye": "eye",
    "both": "both",
}
current_control_mode = CONTROL_MODE_ALIASES.get(os.getenv("CONTROL_MODE", "both").lower(), "both")

status_text = ""
status_until = 0.0

# ---------------------------
# Voice + TTS configuration
# ---------------------------
VOICE_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
VOICE_SAMPLE_RATE = 16000
VOICE_BLOCK_SIZE = 8000
WAKE_WORD = "jarvis"
WAKE_TIMEOUT_SEC = 6.0

voice_enabled = False
voice_status = "Voice: Disabled"
voice_command_queue = queue.Queue()
tts_queue = queue.Queue()
stop_event = threading.Event()
voice_thread = None
tts_thread = None


def find_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def mode_uses_eye(mode):
    return mode in ("eye", "both")


def mode_uses_hand(mode):
    return mode in ("hand", "both")


def mode_label(mode):
    if mode == "hand":
        return "Hand Only"
    if mode == "eye":
        return "Eye Only"
    return "Hand + Eye"


def clamp(value, low, high):
    return max(low, min(high, value))


def landmarks_to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def iris_center(points, iris_indices):
    iris_pts = [points[idx] for idx in iris_indices]
    return np.mean(np.array(iris_pts, dtype=np.float32), axis=0)


def axis_ratio(value, low_point, high_point):
    denom = high_point - low_point
    if abs(denom) < 1e-5:
        return 0.5
    return (value - low_point) / denom


def eye_open_ratio(upper, lower, left_corner, right_corner):
    vertical = abs(upper[1] - lower[1])
    horizontal = max(1.0, abs(right_corner[0] - left_corner[0]))
    return vertical / horizontal


def map_gaze_to_screen(points):
    left_iris_center = iris_center(points, LEFT_IRIS)
    right_iris_center = iris_center(points, RIGHT_IRIS)

    l_left_corner = points[LEFT_EYE_CORNERS[0]]
    l_right_corner = points[LEFT_EYE_CORNERS[1]]
    r_left_corner = points[RIGHT_EYE_CORNERS[0]]
    r_right_corner = points[RIGHT_EYE_CORNERS[1]]

    left_x_ratio = axis_ratio(left_iris_center[0], l_left_corner[0], l_right_corner[0])
    right_x_ratio = axis_ratio(right_iris_center[0], r_left_corner[0], r_right_corner[0])
    x_ratio = (left_x_ratio + right_x_ratio) * 0.5

    l_upper = points[LEFT_EYE_LIDS[0]]
    l_lower = points[LEFT_EYE_LIDS[1]]
    r_upper = points[RIGHT_EYE_LIDS[0]]
    r_lower = points[RIGHT_EYE_LIDS[1]]

    left_y_ratio = axis_ratio(left_iris_center[1], l_upper[1], l_lower[1])
    right_y_ratio = axis_ratio(right_iris_center[1], r_upper[1], r_lower[1])
    y_ratio = (left_y_ratio + right_y_ratio) * 0.5

    x_ratio = float(clamp(x_ratio, 0.0, 1.0))
    y_ratio = float(clamp(y_ratio, 0.0, 1.0))

    if calibration_finished:
        cal_x_ratios = [d['x_ratio'] for d in CALIBRATION_DATA]
        cal_y_ratios = [d['y_ratio'] for d in CALIBRATION_DATA]
        min_x, max_x = min(cal_x_ratios), max(cal_x_ratios)
        min_y, max_y = min(cal_y_ratios), max(cal_y_ratios)
        screen_x = int(np.interp(x_ratio, [min_x, max_x], [0, screen_w - 1]))
        screen_y = int(np.interp(y_ratio, [min_y, max_y], [0, screen_h - 1]))
    else:
        # Fallback to default mapping if not calibrated
        screen_x = int(np.interp(x_ratio, [0.05, 0.95], [0, screen_w - 1]))
        screen_y = int(np.interp(y_ratio, [0.1, 0.9], [0, screen_h - 1]))

    screen_x = int(clamp(screen_x, 0, screen_w - 1))
    screen_y = int(clamp(screen_y, 0, screen_h - 1))

    blink_ratio_left = eye_open_ratio(l_upper, l_lower, l_left_corner, l_right_corner)
    blink_ratio_right = eye_open_ratio(r_upper, r_lower, r_left_corner, r_right_corner)
    blink_ratio = (blink_ratio_left + blink_ratio_right) * 0.5

    return (screen_x, screen_y), (x_ratio, y_ratio), blink_ratio, left_iris_center, right_iris_center


def fingers_up(lm_list):
    return {
        "index": lm_list[8][1] < lm_list[6][1],
        "middle": lm_list[12][1] < lm_list[10][1],
        "ring": lm_list[16][1] < lm_list[14][1],
        "pinky": lm_list[20][1] < lm_list[18][1],
    }


def draw_hand_overlay(img, lm_list):
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, lm_list[start_idx], lm_list[end_idx], (0, 255, 0), 2)
    for point in lm_list:
        cv2.circle(img, point, 4, (255, 0, 255), cv2.FILLED)


def set_status(text, duration=1.5):
    global status_text, status_until
    status_text = text
    status_until = time.time() + duration


def set_volume_percent(percent):
    percent = max(0, min(100, int(percent)))
    target = np.interp(percent, [0, 100], [min_vol, max_vol])
    volume.SetMasterVolumeLevel(float(target), None)
    return percent


def change_volume_by(delta):
    current_level = volume.GetMasterVolumeLevel()
    current_percent = int(np.interp(current_level, [min_vol, max_vol], [0, 100]))
    return set_volume_percent(current_percent + delta)


def save_screenshot(prefix="shot"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    path = os.path.join(screenshot_dir, filename)
    pyautogui.screenshot(path)
    return filename


def enqueue_tts(message):
    if message:
        tts_queue.put(message)


def tts_worker_loop():
    if pyttsx3 is None:
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)

    while not stop_event.is_set():
        try:
            message = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            engine.say(message)
            engine.runAndWait()
        except Exception:
            # Ignore transient TTS errors and keep the assistant alive.
            pass


def parse_intent(command_text):
    text = command_text.lower().strip()

    if any(k in text for k in ["volume", "mute", "unmute", "louder", "quieter"]):
        return "volume_control"
    if any(k in text for k in ["open", "launch", "start app"]):
        return "app_launch"
    if any(k in text for k in ["system", "cpu", "memory", "ram", "time", "date"]):
        return "system_info"
    if any(k in text for k in ["screenshot", "screen shot", "capture screen"]):
        return "screenshot"
    if any(k in text for k in ["play", "pause", "next", "previous", "media", "song"]):
        return "media_control"
    return "unknown"


def optional_llm_fallback(_command_text):
    # Placeholder for future LLM fallback.
    return None


def execute_voice_command(command_text):
    global last_screenshot_time

    text = command_text.lower().strip()
    intent = parse_intent(text)

    if intent == "volume_control":
        if "unmute" in text:
            volume.SetMute(0, None)
            set_status("Unmuted")
            return "System unmuted, sir."

        if "mute" in text:
            volume.SetMute(1, None)
            set_status("Muted")
            return "System muted, sir."

        match = re.search(r"(\d{1,3})", text)
        if match:
            percent = set_volume_percent(int(match.group(1)))
            set_status(f"Volume {percent}%")
            return f"Volume set to {percent} percent, sir."

        if "up" in text or "louder" in text or "increase" in text:
            percent = change_volume_by(10)
            set_status(f"Volume {percent}%")
            return f"Volume increased to {percent} percent, sir."

        if "down" in text or "quieter" in text or "decrease" in text:
            percent = change_volume_by(-10)
            set_status(f"Volume {percent}%")
            return f"Volume decreased to {percent} percent, sir."

        return "Please tell me a volume level, sir."

    if intent == "screenshot":
        if (time.time() - last_screenshot_time) < screenshot_cooldown_sec:
            return "Screenshot command is cooling down, sir."
        filename = save_screenshot(prefix="voice_shot")
        last_screenshot_time = time.time()
        set_status(f"Saved {filename}")
        return "Screenshot captured, sir."

    if intent == "app_launch":
        app_map = {
            "chrome": ["start", "chrome"],
            "notepad": ["start", "notepad"],
            "calculator": ["start", "calc"],
            "paint": ["start", "mspaint"],
        }

        for app_name, cmd in app_map.items():
            if app_name in text:
                subprocess.run(" ".join(cmd), shell=True, check=False)
                set_status(f"Opened {app_name}")
                return f"Opening {app_name}, sir."

        return "I only support Chrome, Notepad, Calculator, and Paint for now, sir."

    if intent == "media_control":
        if "next" in text:
            pyautogui.press("nexttrack")
            set_status("Next Track")
            return "Skipping to next track, sir."
        if "previous" in text:
            pyautogui.press("prevtrack")
            set_status("Previous Track")
            return "Returning to previous track, sir."
        if "play" in text or "pause" in text:
            pyautogui.press("playpause")
            set_status("Play/Pause")
            return "Toggled play and pause, sir."
        return "Please say play, pause, next, or previous, sir."

    if intent == "system_info":
        parts = []

        if psutil is not None:
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            parts.append(f"CPU {cpu:.0f} percent")
            parts.append(f"RAM {ram:.0f} percent")

        now_text = datetime.now().strftime("%I:%M %p")
        parts.append(f"time {now_text}")

        response = " and ".join(parts)
        set_status("System Info")
        return f"Current {response}, sir."

    llm_action = optional_llm_fallback(text)
    if llm_action:
        return llm_action

    return "Command not recognized, sir."


def voice_worker_loop():
    global voice_status

    if sd is None or Model is None or KaldiRecognizer is None:
        voice_status = "Voice: Missing vosk/sounddevice"
        return

    if not os.path.isdir(VOICE_MODEL_PATH):
        voice_status = "Voice: Missing Vosk model folder"
        return

    try:
        model = Model(VOICE_MODEL_PATH)
        recognizer = KaldiRecognizer(model, VOICE_SAMPLE_RATE)
        voice_status = "Voice: Listening for Jarvis"
    except Exception:
        voice_status = "Voice: Model init failed"
        return

    wake_active = False
    wake_until = 0.0

    try:
        with sd.RawInputStream(
            samplerate=VOICE_SAMPLE_RATE,
            blocksize=VOICE_BLOCK_SIZE,
            dtype="int16",
            channels=1,
        ) as stream:
            while not stop_event.is_set():
                data, _ = stream.read(VOICE_BLOCK_SIZE)
                if not recognizer.AcceptWaveform(data):
                    continue

                result = recognizer.Result()
                text = ""
                try:
                    import json

                    text = json.loads(result).get("text", "").strip().lower()
                except Exception:
                    text = ""

                if not text:
                    continue

                now = time.time()

                if WAKE_WORD in text:
                    after_wake = text.split(WAKE_WORD, 1)[1].strip(" ,")
                    wake_active = True
                    wake_until = now + WAKE_TIMEOUT_SEC
                    voice_status = "Voice: Wake confirmed"

                    if after_wake:
                        voice_command_queue.put(after_wake)
                        wake_active = False
                        voice_status = "Voice: Listening for Jarvis"
                    continue

                if wake_active and now <= wake_until:
                    voice_command_queue.put(text)
                    wake_active = False
                    voice_status = "Voice: Listening for Jarvis"
                elif wake_active and now > wake_until:
                    wake_active = False
                    voice_status = "Voice: Listening for Jarvis"
    except Exception:
        voice_status = "Voice: Microphone error"


# Start assistant worker threads.
if pyttsx3 is not None:
    tts_thread = threading.Thread(target=tts_worker_loop, daemon=True)
    tts_thread.start()

if sd is not None and Model is not None and KaldiRecognizer is not None:
    voice_enabled = True
    voice_thread = threading.Thread(target=voice_worker_loop, daemon=True)
    voice_thread.start()
else:
    voice_status = "Voice: Install vosk + sounddevice"

# ---------------------------
# Main loop (gesture + voice)
# ---------------------------
while True:
    success, img = cap.read()
    if not success:
        continue

    now = time.time()
    frame_delta = now - last_frame_time
    fps = int(1.0 / frame_delta) if frame_delta > 0 else 0
    last_frame_time = now

    vol_percent = None
    hand_detected = False
    gaze_point = None
    blink_ratio = None

    # Execute queued voice commands in the main thread.
    while not voice_command_queue.empty():
        command = voice_command_queue.get()
        reply = execute_voice_command(command)
        enqueue_tts(reply)

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    use_eye = mode_uses_eye(current_control_mode)
    use_hand = mode_uses_hand(current_control_mode)

    # Eye tracking (iris -> cursor).
    if use_eye:
        face_result = face_detector.detect(mp_image)
    else:
        face_result = None

    if use_eye and face_result.face_landmarks:
        h, w, _ = img.shape
        face_points = landmarks_to_pixels(face_result.face_landmarks[0], w, h)
        mapped_point, raw_ratios, blink_ratio, left_iris_center, right_iris_center = map_gaze_to_screen(face_points)

        if is_calibrating:
            point_to_cal = CALIBRATION_POINTS[calibration_step]
            cv2.circle(img, point_to_cal, 20, (0, 0, 255), 3)
            cv2.putText(img, f"Look at the circle and press SPACE ({calibration_step + 1}/{len(CALIBRATION_POINTS)})", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if gaze_smooth is None:
                gaze_smooth = np.array(mapped_point, dtype=np.float32)
                last_gaze_point = gaze_smooth
            else:
                # Dynamic smoothing
                move_dist = np.linalg.norm(np.array(mapped_point) - last_gaze_point)
                alpha = gaze_alpha_fast if move_dist > gaze_fast_threshold else gaze_alpha_slow
                gaze_smooth = (1 - alpha) * gaze_smooth + alpha * np.array(mapped_point, dtype=np.float32)

            gaze_point = (int(gaze_smooth[0]), int(gaze_smooth[1]))
            last_gaze_point = np.array(gaze_point)
            try:
                pyautogui.moveTo(gaze_point[0], gaze_point[1], duration=0)
            except Exception:
                pass
    else:
        raw_ratios = None


        if blink_ratio is not None:
            if blink_ratio_smooth is None:
                blink_ratio_smooth = blink_ratio
            else:
                blink_ratio_smooth = (1 - blink_alpha) * blink_ratio_smooth + blink_alpha * blink_ratio

            if blink_ratio_smooth < blink_click_threshold:
                # Blink detected
                if blink_state == "NO_BLINK":
                    blink_state = "BLINK_1"
                    last_blink_time = now
                elif blink_state == "BLINK_1" and (now - last_blink_time) < double_blink_time_sec:
                    blink_state = "BLINK_2"
            else:
                # No blink
                if blink_state == "BLINK_1" and (now - last_blink_time) > blink_click_cooldown_sec:
                    if (now - last_blink_click_time) > blink_click_cooldown_sec:
                        pyautogui.click()
                        last_blink_click_time = now
                        set_status("Blink Click")
                    blink_state = "NO_BLINK"
                elif blink_state == "BLINK_2":
                     if (now - last_blink_click_time) > blink_click_cooldown_sec:
                        pyautogui.doubleClick()
                        last_blink_click_time = now
                        set_status("Double Blink Click")
                     blink_state = "NO_BLINK"

        # Reset blink state if too much time has passed
        if blink_state != "NO_BLINK" and (now - last_blink_time) > double_blink_time_sec + 0.1:
            blink_state = "NO_BLINK"

        cv2.circle(img, tuple(left_iris_center.astype(int)), 4, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, tuple(right_iris_center.astype(int)), 4, (255, 255, 0), cv2.FILLED)

    if use_hand:
        result = hand_detector.detect(mp_image)
    else:
        result = None

    if result is not None and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            hand_detected = True
            lm_list = []
            for lm in hand_landmarks:
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            thumb = lm_list[4]
            index = lm_list[8]
            up = fingers_up(lm_list)
            is_fist = not (up["index"] or up["middle"] or up["ring"] or up["pinky"])

            # Hold fist to lock/unlock all gesture actions.
            if is_fist:
                if fist_hold_start is None:
                    fist_hold_start = now
                elif (
                    (now - fist_hold_start) > fist_hold_sec
                    and (now - last_lock_toggle_time) > lock_cooldown_sec
                ):
                    controls_locked = not controls_locked
                    last_lock_toggle_time = now
                    fist_hold_start = now
                    lock_text = "Controls locked, sir." if controls_locked else "Controls unlocked, sir."
                    set_status("Controls Locked" if controls_locked else "Controls Unlocked")
                    enqueue_tts(lock_text)
            else:
                fist_hold_start = None

            if controls_locked:
                draw_hand_overlay(img, lm_list)
                continue

            # Gesture volume from thumb-index distance.
            distance = find_distance(thumb, index)
            target_vol = np.interp(distance, [30, 200], [min_vol, max_vol])
            vol_percent = int(np.interp(distance, [30, 200], [0, 100]))
            vol_percent = max(0, min(100, vol_percent))

            if smoothed_vol is None:
                smoothed_vol = target_vol
            else:
                smoothed_vol = (1 - volume_alpha) * smoothed_vol + volume_alpha * target_vol
            volume.SetMasterVolumeLevel(float(smoothed_vol), None)

            # Pinch thumb and index to toggle mute/unmute.
            if distance < 25 and (now - last_mute_toggle_time) > mute_cooldown_sec:
                current_mute = int(volume.GetMute())
                new_mute = 0 if current_mute == 1 else 1
                volume.SetMute(new_mute, None)
                last_mute_toggle_time = now
                set_status("Muted" if new_mute == 1 else "Unmuted")
                enqueue_tts("Muted, sir." if new_mute == 1 else "Unmuted, sir.")

            # Raise 4 fingers to take screenshot.
            if (
                up["index"]
                and up["middle"]
                and up["ring"]
                and up["pinky"]
                and (now - last_screenshot_time) > screenshot_cooldown_sec
            ):
                filename = save_screenshot(prefix="gesture_shot")
                last_screenshot_time = now
                set_status(f"Saved {filename}")
                enqueue_tts("Screenshot captured, sir.")

            cv2.line(img, thumb, index, (0, 255, 0), 3)
            cv2.circle(img, thumb, 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, index, 8, (255, 0, 255), cv2.FILLED)

            draw_hand_overlay(img, lm_list)

    if use_hand and not hand_detected:
        fist_hold_start = None

    if vol_percent is not None:
        cv2.putText(
            img,
            f"Volume: {vol_percent}%",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        bar_top = 120
        bar_bottom = 320
        bar_left = 20
        bar_right = 50
        fill_y = int(np.interp(vol_percent, [0, 100], [bar_bottom, bar_top]))
        cv2.rectangle(img, (bar_left, bar_top), (bar_right, bar_bottom), (255, 255, 255), 2)
        cv2.rectangle(img, (bar_left, fill_y), (bar_right, bar_bottom), (0, 255, 0), cv2.FILLED)
    else:
        cv2.putText(
            img,
            "Volume Control: Show thumb + index",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

    if now < status_until:
        cv2.putText(
            img,
            status_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

    cv2.putText(
        img,
        f"FPS: {fps}",
        (20, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    lock_label = "LOCKED" if controls_locked else "ACTIVE"
    lock_color = (0, 0, 255) if controls_locked else (0, 255, 0)
    cv2.putText(
        img,
        f"Controls: {lock_label}",
        (20, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        lock_color,
        2,
    )

    cv2.putText(
        img,
        voice_status,
        (20, 440),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        img,
        "Voice: Say Jarvis then command",
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        img,
        "Eye Move=Cursor  Blink=Click  Pinch=Mute  4 Fingers=Shot",
        (20, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        img,
        "Press 'c' to start calibration",
        (20, 530),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        img,
        f"Mode: {mode_label(current_control_mode)} (1=Hand, 2=Eye, 3=Both)",
        (20, 560),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 255, 140),
        2,
    )

    if gaze_point is not None:
        cv2.putText(
            img,
            f"Gaze Cursor: ({gaze_point[0]}, {gaze_point[1]})",
            (20, 590),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 220, 255),
            2,
        )
        if blink_ratio is not None:
            cv2.putText(
                img,
                f"Eye Open Ratio: {blink_ratio:.2f}",
                (20, 620),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (120, 220, 255),
                2,
            )
    else:
        cv2.putText(
            img,
            "Eye Tracking: Face not detected / mode off",
            (20, 590),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )

    cv2.imshow("Jarvis Gesture + Voice Control", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in (ord("1"), ord("2"), ord("3")):
        current_control_mode = CONTROL_MODE_ALIASES.get(chr(key), "both")
        set_status(f"Mode: {mode_label(current_control_mode)}")
    elif key == ord("c"):
        start_calibration()
    elif key == ord(" ") and is_calibrating:
        if raw_ratios:
            CALIBRATION_DATA.append({
                "screen_point": CALIBRATION_POINTS[calibration_step],
                "x_ratio": raw_ratios[0],
                "y_ratio": raw_ratios[1],
            })
            calibration_step += 1
            if calibration_step >= len(CALIBRATION_POINTS):
                is_calibrating = False
                calibration_finished = True
                set_status("Calibration Complete!", duration=3.0)
    
stop_event.set()
if voice_thread is not None:
    voice_thread.join(timeout=1.0)
if tts_thread is not None:
    tts_thread.join(timeout=1.0)

hand_detector.close()
face_detector.close()
cap.release()
cv2.destroyAllWindows()