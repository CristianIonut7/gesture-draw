import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Descarc modelul MediaPipe... (o singura data, ~25MB)")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model descarcat!")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

COLORS = [
    ((255, 255, 255), "Alb"),
    ((0,   0,   255), "Rosu"),
    ((0,   255,   0), "Verde"),
    ((255,   0,   0), "Albastru"),
    ((0,   255, 255), "Galben"),
    ((255,   0, 255), "Mov"),
]
color_index = 0

# Rezolutia de afisare (fereastra mare)
DISPLAY_W, DISPLAY_H = 1280, 720

canvas = None  # initializat la primul frame pe dimensiunea camerei
prev_x, prev_y = 0, 0
fist_frames = 0
open_frames = 0
FIST_THRESHOLD = 35
OPEN_THRESHOLD = 25

def is_fist(landmarks):
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    for tip, mcp in zip(tips, mcps):
        if landmarks[tip].y < landmarks[mcp].y:
            return False
    return True

def is_open_hand(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y > landmarks[pip].y:
            return False
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist > 0.1

def get_finger_state(landmarks, w, h):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    ix = int(index_tip.x * w)
    iy = int(index_tip.y * h)
    dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    pen_up = dist < 0.07
    return ix, iy, pen_up

def draw_color_palette(frame, current_index):
    x_start = frame.shape[1] - 30
    for i, (color, _) in enumerate(COLORS):
        y = 20 + i * 28
        cv2.circle(frame, (x_start, y), 10, color, -1)
        if i == current_index:
            cv2.circle(frame, (x_start, y), 13, (255, 255, 255), 2)

cap = cv2.VideoCapture(1)
# Camera rămâne la rezolutia ei nativa (ex: 640x480) - procesare rapida
# Fereastra va fi scalata la DISPLAY_W x DISPLAY_H
print("Camera pornita!")
print("Index = desenezi | Ciupire = pen up | Palma = culoare | Pumn = clear")
print("Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # dimensiunea reala a camerei (ex: 480x640)

    # Initializeaza canvas la dimensiunea camerei (o singura data)
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    current_color = COLORS[color_index][0]
    current_color_name = COLORS[color_index][1]

    # MediaPipe ruleaza pe frame-ul mic = rapid
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture_label = ""

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        for start_idx, end_idx in HAND_CONNECTIONS:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            sx, sy = int(start.x * w), int(start.y * h)
            ex, ey = int(end.x * w), int(end.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), (80, 80, 80), 1)

        if is_fist(landmarks):
            fist_frames += 1
            open_frames = 0
            prev_x, prev_y = 0, 0
            gesture_label = "PUMN - clear"

            progress = min(fist_frames / FIST_THRESHOLD, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 + 100, h - 20), (40, 40, 40), -1)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 - 100 + bar_w, h - 20), (0, 0, 220), -1)
            cv2.putText(frame, "CLEAR...", (w//2 - 35, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if fist_frames >= FIST_THRESHOLD:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                fist_frames = 0
                print("Canvas cleared!")

        elif is_open_hand(landmarks):
            open_frames += 1
            fist_frames = 0
            prev_x, prev_y = 0, 0
            gesture_label = "PALMA - culoare"

            progress = min(open_frames / OPEN_THRESHOLD, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 + 100, h - 20), (40, 40, 40), -1)
            cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 - 100 + bar_w, h - 20), current_color, -1)
            cv2.putText(frame, "CULOARE...", (w//2 - 40, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if open_frames >= OPEN_THRESHOLD:
                color_index = (color_index + 1) % len(COLORS)
                open_frames = 0
                print(f"Culoare schimbata: {COLORS[color_index][1]}")

        else:
            fist_frames = 0
            open_frames = 0
            ix, iy, pen_up = get_finger_state(landmarks, w, h)

            if pen_up:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
                gesture_label = "PEN UP"
            else:
                cv2.circle(frame, (ix, iy), 8, current_color, -1)
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, 4)
                prev_x, prev_y = ix, iy
                gesture_label = f"DESEN ({current_color_name})"
    else:
        prev_x, prev_y = 0, 0
        fist_frames = 0
        open_frames = 0

    # Suprapune canvas pe frame (ambele la dimensiunea camerei)
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # Scalare pentru afisare (doar la final, nu afecteaza procesarea)
    display = cv2.resize(combined, (DISPLAY_W, DISPLAY_H))

    # UI pe frame-ul scalat
    draw_color_palette(display, color_index)
    cv2.putText(display, "Index=desen | Ciupire=pen up | Palma=culoare | Pumn=clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    if gesture_label:
        cv2.putText(display, gesture_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

    cv2.imshow("GestureWar - Canvas", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()