import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from canvas import (
    MODEL_PATH, ensure_hand_model,
    HAND_CONNECTIONS, is_fist, get_finger_state,
    DISPLAY_W, DISPLAY_H,
)


PEN_COLOR_P1 = (255, 200, 100)   # blue-cyan
PEN_COLOR_P2 = (100, 180, 255)   # orange
FIST_THRESHOLD = 15


class DuelCanvas:
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

    def reset_canvases(self):
        if self.canvas_p1 is not None:
            self.canvas_p1 = np.zeros_like(self.canvas_p1)
            self.canvas_p2 = np.zeros_like(self.canvas_p2)
        self.prev_p1 = (0, 0)
        self.prev_p2 = (0, 0)
        self.fist_frames_p1 = 0
        self.fist_frames_p2 = 0

    def update(self, drawing_enabled=True, allow_clear=True):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        half_w = w // 2

        if self.canvas_p1 is None:
            self.canvas_p1 = np.zeros((h, half_w, 3), dtype=np.uint8)
            self.canvas_p2 = np.zeros((h, w - half_w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        p1_seen = False
        p2_seen = False
        p1_fist_this_frame = False
        p2_fist_this_frame = False

        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                avg_x = (landmarks[0].x + landmarks[9].x) / 2
                player = 1 if avg_x < 0.5 else 2

                ix_full = int(landmarks[8].x * w)
                iy_full = int(landmarks[8].y * h)

                skel_color = PEN_COLOR_P1 if player == 1 else PEN_COLOR_P2
                for s, e in HAND_CONNECTIONS:
                    p1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
                    p2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
                    cv2.line(frame, p1, p2, tuple(c // 2 for c in skel_color), 1)

                fist = is_fist(landmarks)
                _, _, pen_up = get_finger_state(landmarks, w, h)

                if player == 1:
                    p1_seen = True
                    cx = max(0, min(half_w - 1, ix_full))
                    cy = max(0, min(h - 1, iy_full))

                    if allow_clear and fist:
                        p1_fist_this_frame = True
                        self.fist_frames_p1 += 1
                        self.prev_p1 = (0, 0)
                        progress = min(self.fist_frames_p1 / FIST_THRESHOLD, 1.0)
                        self._draw_clear_progress(frame, half_w // 2, h, progress, PEN_COLOR_P1, "CLEAR")
                        if self.fist_frames_p1 >= FIST_THRESHOLD:
                            self.canvas_p1 = np.zeros_like(self.canvas_p1)
                            self.fist_frames_p1 = 0
                    elif pen_up or not drawing_enabled:
                        self.prev_p1 = (0, 0)
                        cv2.circle(frame, (ix_full, iy_full), 10, (0, 0, 255), -1)
                    else:
                        cv2.circle(frame, (ix_full, iy_full), 8, PEN_COLOR_P1, -1)
                        if self.prev_p1 != (0, 0):
                            cv2.line(self.canvas_p1, self.prev_p1, (cx, cy),
                                     (255, 255, 255), 4)
                        self.prev_p1 = (cx, cy)

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

        if not p1_seen:
            self.prev_p1 = (0, 0)
        if not p2_seen:
            self.prev_p2 = (0, 0)
        if not p1_fist_this_frame:
            self.fist_frames_p1 = 0
        if not p2_fist_this_frame:
            self.fist_frames_p2 = 0

        composite = frame.copy()

        canvas_p1_gray = cv2.cvtColor(self.canvas_p1, cv2.COLOR_BGR2GRAY)
        _, mask_p1 = cv2.threshold(canvas_p1_gray, 10, 255, cv2.THRESH_BINARY)
        colored_p1 = np.zeros_like(self.canvas_p1)
        colored_p1[mask_p1 > 0] = PEN_COLOR_P1
        left_part = composite[:, :half_w]
        mask_inv = cv2.bitwise_not(mask_p1)
        left_bg = cv2.bitwise_and(left_part, left_part, mask=mask_inv)
        left_fg = cv2.bitwise_and(colored_p1, colored_p1, mask=mask_p1)
        composite[:, :half_w] = cv2.add(left_bg, left_fg)

        canvas_p2_gray = cv2.cvtColor(self.canvas_p2, cv2.COLOR_BGR2GRAY)
        _, mask_p2 = cv2.threshold(canvas_p2_gray, 10, 255, cv2.THRESH_BINARY)
        colored_p2 = np.zeros_like(self.canvas_p2)
        colored_p2[mask_p2 > 0] = PEN_COLOR_P2
        right_part = composite[:, half_w:]
        mask_inv2 = cv2.bitwise_not(mask_p2)
        right_bg = cv2.bitwise_and(right_part, right_part, mask=mask_inv2)
        right_fg = cv2.bitwise_and(colored_p2, colored_p2, mask=mask_p2)
        composite[:, half_w:] = cv2.add(right_bg, right_fg)

        cv2.line(composite, (half_w, 0), (half_w, h), (220, 220, 220), 2)

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

    def _draw_clear_progress(self, frame, center_x, h, progress, color, label):
        bar_w = int(120 * progress)
        cv2.rectangle(frame, (center_x - 60, h - 70), (center_x + 60, h - 50), (40, 40, 40), -1)
        cv2.rectangle(frame, (center_x - 60, h - 70), (center_x - 60 + bar_w, h - 50), color, -1)
        cv2.putText(frame, label, (center_x - 25, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def close(self):
        self.cap.release()
        self.detector.close()