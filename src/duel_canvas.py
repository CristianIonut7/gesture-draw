"""A shared drawing canvas for two players using hand tracking."""

# Disable oneDNN optimizations and TensorFlow logging for cleaner output
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2 # OpenCV is used for video capture and drawing on the canvas
import numpy as np # NumPy is used for array manipulations and creating the canvas
import mediapipe as mp # MediaPipe is used for hand tracking and landmark detection
from mediapipe.tasks import python as mp_python # MediaPipe Tasks API for hand landmark detection
from mediapipe.tasks.python import vision # Vision API for hand landmark detection

# Import helper functions and constants from the canvas module
from canvas import (
    MODEL_PATH, ensure_hand_model,
    HAND_CONNECTIONS, is_fist, get_finger_state,
    DISPLAY_W, DISPLAY_H,
)

# Define colors and thresholds
PEN_COLOR_P1 = (255, 200, 100)
PEN_COLOR_P2 = (100, 180, 255)
FIST_THRESHOLD = 15


class DuelCanvas:
    # Constructor
    def __init__(self):
        ensure_hand_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)

        self.canvas_p1 = None
        self.canvas_p2 = None
        self.prev_p1 = (0, 0)
        self.prev_p2 = (0, 0)
        self.fist_frames_p1 = 0
        self.fist_frames_p2 = 0
        self.frame_counter = 0
        self.last_mp_result = None

    # Resets the canvases for both players
    def reset_canvases(self):
        if self.canvas_p1 is not None:
            self.canvas_p1 = np.zeros_like(self.canvas_p1)
            self.canvas_p2 = np.zeros_like(self.canvas_p2)
        self.prev_p1 = (0, 0)
        self.prev_p2 = (0, 0)
        self.fist_frames_p1 = 0
        self.fist_frames_p2 = 0

    def update(self, drawing_enabled=True, allow_clear=True):
        # Capture a frame
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        half_w = w // 2

        # Initialize the player canvases if they haven't been created yet
        if self.canvas_p1 is None:
            self.canvas_p1 = np.zeros((h, half_w, 3), dtype=np.uint8)
            self.canvas_p2 = np.zeros((h, w - half_w, 3), dtype=np.uint8)

        # Only run the hand landmark detection every 2 frames to improve performance, and reuse the last result in between
        self.frame_counter += 1
        if self.frame_counter % 2 == 0 or self.last_mp_result is None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            self.last_mp_result = result
        else:
            result = self.last_mp_result

        # Initialize flags
        p1_seen = False
        p2_seen = False
        p1_fist_this_frame = False
        p2_fist_this_frame = False

        if result.hand_landmarks: # If any hands are detected
            # Process each detected hand
            for landmarks in result.hand_landmarks:
                # Determine which player the hand belongs to
                avg_x = (landmarks[0].x + landmarks[9].x) / 2
                player = 1 if avg_x < 0.5 else 2
                
                # Get the coordinates of the index fingertip
                ix_full = int(landmarks[8].x * w)
                iy_full = int(landmarks[8].y * h)

                # Draw the skeleton for visualization
                skel_color = PEN_COLOR_P1 if player == 1 else PEN_COLOR_P2
                for s, e in HAND_CONNECTIONS:
                    p1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
                    p2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
                    cv2.line(frame, p1, p2, tuple(c // 2 for c in skel_color), 1)

                # Check for fist and pen up states
                fist = is_fist(landmarks)
                _, _, pen_up = get_finger_state(landmarks, w, h)

                if player == 1:
                    p1_seen = True
                    cx = max(0, min(half_w - 1, ix_full))
                    cy = max(0, min(h - 1, iy_full))
                    
                    # If the player is making a fist and clearing is allowed
                    if allow_clear and fist:
                        p1_fist_this_frame = True
                        self.fist_frames_p1 += 1
                        self.prev_p1 = (0, 0)
                        progress = min(self.fist_frames_p1 / FIST_THRESHOLD, 1.0)
                        self._draw_clear_progress(frame, half_w // 2, h, progress, PEN_COLOR_P1, "CLEAR")
                        if self.fist_frames_p1 >= FIST_THRESHOLD:
                            self.canvas_p1 = np.zeros_like(self.canvas_p1)
                            self.fist_frames_p1 = 0
                    # If the pen is up or drawing is disabled
                    elif pen_up or not drawing_enabled:
                        self.prev_p1 = (0, 0)
                        cv2.circle(frame, (ix_full, iy_full), 10, (0, 0, 255), -1)
                    # Otherwise, draw on the canvas
                    else:
                        cv2.circle(frame, (ix_full, iy_full), 8, PEN_COLOR_P1, -1)
                        if self.prev_p1 != (0, 0):
                            cv2.line(self.canvas_p1, self.prev_p1, (cx, cy),
                                     (255, 255, 255), 4)
                        self.prev_p1 = (cx, cy)
                # Player 2 processing is similar but for the right half of the screen
                else:
                    p2_seen = True
                    cx = max(0, min(w - half_w - 1, ix_full - half_w))
                    cy = max(0, min(h - 1, iy_full))

                    if allow_clear and fist:
                        p2_fist_this_frame = True
                        self.fist_frames_p2 += 1
                        self.prev_p2 = (0, 0)
                        progress = min(self.fist_frames_p2 / FIST_THRESHOLD, 1.0)
                        self._draw_clear_progress(frame, half_w + (w - half_w) // 2, h, progress,
                                                   PEN_COLOR_P2, "CLEAR")
                        if self.fist_frames_p2 >= FIST_THRESHOLD:
                            self.canvas_p2 = np.zeros_like(self.canvas_p2)
                            self.fist_frames_p2 = 0
                    elif pen_up or not drawing_enabled:
                        self.prev_p2 = (0, 0)
                        cv2.circle(frame, (ix_full, iy_full), 10, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame, (ix_full, iy_full), 8, PEN_COLOR_P2, -1)
                        if self.prev_p2 != (0, 0):
                            cv2.line(self.canvas_p2, self.prev_p2, (cx, cy),
                                     (255, 255, 255), 4)
                        self.prev_p2 = (cx, cy)

        # If a player is not seen, reset their previous position
        if not p1_seen:
            self.prev_p1 = (0, 0)
        if not p2_seen:
            self.prev_p2 = (0, 0)
        if not p1_fist_this_frame:
            self.fist_frames_p1 = 0
        if not p2_fist_this_frame:
            self.fist_frames_p2 = 0

        # Create a composite image of the frame and the canvases for display
        composite = frame.copy()
        canvas_p1_mask = self.canvas_p1[:, :, 0] > 10
        composite[:, :half_w][canvas_p1_mask] = PEN_COLOR_P1
        canvas_p2_mask = self.canvas_p2[:, :, 0] > 10
        composite[:, half_w:][canvas_p2_mask] = PEN_COLOR_P2

        # Draw a dividing line between the two halves
        cv2.line(composite, (half_w, 0), (half_w, h), (220, 220, 220), 2)

        # Resize the composite image to fit the display size
        display = cv2.resize(composite, (DISPLAY_W, DISPLAY_H))

        return {
            "display": display,
            "raw_canvas_p1": self.canvas_p1.copy(),
            "raw_canvas_p2": self.canvas_p2.copy(),
            "p1_active": p1_seen,
            "p2_active": p2_seen,
            "frame_w": w,
            "frame_h": h,
        }

    # Draws a progress bar on the frame
    def _draw_clear_progress(self, frame, center_x, h, progress, color, label):
        bar_w = int(120 * progress)
        cv2.rectangle(frame, (center_x - 60, h - 70), (center_x + 60, h - 50), (40, 40, 40), -1)
        cv2.rectangle(frame, (center_x - 60, h - 70), (center_x - 60 + bar_w, h - 50), color, -1)
        cv2.putText(frame, label, (center_x - 25, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Release resources when done
    def close(self):
        self.cap.release()
        self.detector.close()