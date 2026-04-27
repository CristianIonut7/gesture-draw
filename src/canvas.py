"""A virtual drawing canvas controlled by hand gestures using MediaPipe."""

# Disable oneDNN optimizations and TensorFlow logging for cleaner output
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Standard imports
import cv2 # OpenCV for video capture and drawing
import numpy as np # NumPy for array manipulations
import urllib.request # For downloading the MediaPipe model if not present
import mediapipe as mp # MediaPipe for hand tracking
from mediapipe.tasks import python as mp_python # MediaPipe tasks API
from mediapipe.tasks.python import vision   # MediaPipe vision tasks for hand landmark detection

MODEL_PATH = "models/hand_landmarker.task" # Local path to the MediaPipe hand landmark model
# URL to download the MediaPipe hand landmark model if not present
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

DISPLAY_W, DISPLAY_H = 1280, 720
FIST_THRESHOLD = 15 # Number of consecutive frames with a fist gesture
OPEN_THRESHOLD = 20 # Number of consecutive frames with an open hand gesture

# Define hand connections for drawing the skeleton (based on MediaPipe's hand landmark model)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# Define a set of colors
COLORS = [
    ((255, 255, 255), "White"),
    ((0,   0,   255), "Red"),
    ((0,   255,   0), "Green"),
    ((255,   0,   0), "Blue"),
    ((0,   255, 255), "Yellow"),
    ((255,   0, 255), "Magenta"),
]

# Ensure the MediaPipe model is downloaded and available locally
def ensure_hand_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe model... (one-time, ~25MB)")
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# is_fist checks if the hand landmarks correspond to a fist gesture
def is_fist(landmarks):
    tips = [8, 12, 16, 20] # Index, middle, ring, pinky fingertips
    mcps = [5,  9, 13, 17] # Corresponding MCP joints for the fingers
    for tip, mcp in zip(tips, mcps): # Check if each fingertip is below its corresponding MCP joint
        if landmarks[tip].y < landmarks[mcp].y:
            return False
    return True


# is_open_hand checks if the hand landmarks correspond to an open hand gesture
def is_open_hand(landmarks):
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky fingertips
    pips = [6, 10, 14, 18]  # Corresponding PIP joints for the fingers  
    for tip, pip in zip(tips, pips): # Check if each fingertip is above its corresponding PIP joint
        if landmarks[tip].y > landmarks[pip].y:
            return False
    thumb_tip = landmarks[4] # Thumb tip
    index_tip = landmarks[8] # Index fingertip
    dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) # Check if thumb and index fingertips are sufficiently apart to confirm open hand
    return dist > 0.1

# get_finger_state calculates the position of the index fingertip and determines if the pen is up or down
def get_finger_state(landmarks, w, h):
    index_tip = landmarks[8] # Index fingertip
    thumb_tip = landmarks[4] # Thumb tip
    ix = int(index_tip.x * w)
    iy = int(index_tip.y * h)
    dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) # Distance between thumb and index fingertips
    pen_up = dist < 0.07
    return ix, iy, pen_up

class GestureCanvas:

    # Constructor
    def __init__(self, num_hands=1, allow_color_change=True, allow_clear=True):
        ensure_hand_model() 
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH) # Set up MediaPipe hand landmark detection options
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands, # Allow detection of one or two hands
            min_hand_detection_confidence=0.4, # Minimum confidence for hand detection
            min_hand_presence_confidence=0.5, # Minimum confidence for hand presence
            min_tracking_confidence=0.5 # Minimum confidence for hand tracking
        )
        self.detector = vision.HandLandmarker.create_from_options(options) # Create the hand landmark detector
        self.cap = cv2.VideoCapture(0) # Start video capture from the default webcam

        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.fist_frames = 0
        self.open_frames = 0
        self.color_index = 0
        self.allow_color_change = True
        self.allow_clear = True
        self.set_modes(allow_color_change, allow_clear)
        self.just_cleared = False
        self.open_hand_triggered = False

    #set_modes allows enabling or disabling the color change and canvas clear functionalities
    def set_modes(self, allow_color_change=True, allow_clear=True):
        self.allow_color_change = allow_color_change
        self.allow_clear = allow_clear

    # reset_canvas clears the drawing canvas
    def reset_canvas(self):
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.prev_x, self.prev_y = 0, 0
        self.fist_frames = 0
        self.open_frames = 0

    def get_current_color(self):
        return COLORS[self.color_index][0]

    def get_current_color_name(self):
        return COLORS[self.color_index][1]

    def update(self):
        ret, frame = self.cap.read() # Capture a frame from the webcam
        if not ret:
            return None

        frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
        h, w, _ = frame.shape

        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        gesture_label = "" # Label to describe the current state
        current_color = self.get_current_color()
        self.just_cleared = False
        self.open_hand_triggered = False

        open_hand_active = False

        # Process detected hand landmarks and determine gestures
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Draw hand skeleton on the frame
            for s, e in HAND_CONNECTIONS:
                p1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
                p2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
                cv2.line(frame, p1, p2, (80, 80, 80), 1)

            # Handle fist gesture for clearing the canvas
            if self.allow_clear and is_fist(landmarks):
                self.fist_frames += 1
                self.open_frames = 0
                self.prev_x, self.prev_y = 0, 0
                gesture_label = "FIST - clear"

                progress = min(self.fist_frames / FIST_THRESHOLD, 1.0)
                self._draw_progress_bar(frame, w, h, progress, (0, 0, 220), "CLEAR...")

                if self.fist_frames >= FIST_THRESHOLD:
                    self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    self.fist_frames = 0
                    self.just_cleared = True

            # Handle open hand gesture for color change or other action
            elif is_open_hand(landmarks):
                open_hand_active = True
                self.open_frames += 1
                self.fist_frames = 0
                self.prev_x, self.prev_y = 0, 0

                progress = min(self.open_frames / OPEN_THRESHOLD, 1.0)

                if self.allow_color_change:
                    gesture_label = "OPEN - color"
                    self._draw_progress_bar(frame, w, h, progress, current_color, "COLOR...")
                    if self.open_frames >= OPEN_THRESHOLD:
                        self.color_index = (self.color_index + 1) % len(COLORS)
                        self.open_frames = 0
                        self.open_hand_triggered = True
                else:
                    gesture_label = "OPEN HAND"
                    if self.open_frames >= OPEN_THRESHOLD:
                        self.open_hand_triggered = True
                        self.open_frames = 0

            # Handle drawing with the index finger when pen is down
            else:
                self.fist_frames = 0
                self.open_frames = 0
                ix, iy, pen_up = get_finger_state(landmarks, w, h)

                if pen_up:
                    self.prev_x, self.prev_y = 0, 0
                    cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
                    gesture_label = "PEN UP"
                else:
                    cv2.circle(frame, (ix, iy), 8, current_color, -1)
                    if self.prev_x != 0 and self.prev_y != 0:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (ix, iy),
                                 current_color, 4)
                    self.prev_x, self.prev_y = ix, iy
                    gesture_label = f"DRAWING ({self.get_current_color_name()})"
        # No hand is detected
        else:
            self.prev_x, self.prev_y = 0, 0
            self.fist_frames = 0
            self.open_frames = 0

        # Composite the canvas with the current video frame
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        composite = cv2.add(frame_bg, canvas_fg)

        # Draw the color palette on the composite image
        display = cv2.resize(composite, (DISPLAY_W, DISPLAY_H))

        return {
            "display": display,
            "gesture_label": gesture_label,
            "current_color": current_color,
            "current_color_name": self.get_current_color_name(),
            "raw_canvas": self.canvas.copy(),
            "frame_w": w,
            "frame_h": h,
            "open_hand_active": open_hand_active,
            "open_hand_progress": min(self.open_frames / OPEN_THRESHOLD, 1.0) if open_hand_active else 0.0,
            "open_hand_triggered": self.open_hand_triggered,
        }

    # _draw_progress_bar draws a progress bar
    def _draw_progress_bar(self, frame, w, h, progress, color, label):
        bar_w = int(200 * progress)
        cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 + 100, h - 20), (40, 40, 40), -1)
        cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 - 100 + bar_w, h - 20), color, -1)
        cv2.putText(frame, label, (w//2 - 40, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # draws the color palette on the display
    def draw_color_palette(self, display):
        x_start = display.shape[1] - 30
        for i, (color, _) in enumerate(COLORS):
            y = 20 + i * 28
            cv2.circle(display, (x_start, y), 10, color, -1)
            if i == self.color_index:
                cv2.circle(display, (x_start, y), 13, (255, 255, 255), 2)

    # close releases the video capture and closes the MediaPipe detector
    def close(self):
        self.cap.release()
        self.detector.close()