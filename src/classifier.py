"""
classifier.py
Wrapper pentru modelul Quick Draw antrenat.
Primeste un canvas (imagine numpy) si returneaza ce crede ca e desenat.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import json
import tensorflow as tf

MODEL_PATH = "models/quickdraw_model.h5"
LABELS_PATH = "models/labels.json"
IMG_SIZE = 64


class DrawingClassifier:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        print("Incarc modelul de clasificare...")
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        print(f"Model incarcat. {len(self.labels)} categorii disponibile.")

    def preprocess(self, canvas):
        """
        Pregateste canvas-ul pentru model:
        1. Convert grayscale daca e color
        2. Crop la bounding box-ul desenului (centrat)
        3. Resize la IMG_SIZE x IMG_SIZE
        4. Normalize 0-1
        """
        # Grayscale
        if len(canvas.shape) == 3:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:
            gray = canvas.copy()

        if cv2.countNonZero(gray) < 50:
            return None

        # Bounding box al desenului
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)

        padding = 20
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(gray.shape[1], x + w + padding)
        y_max = min(gray.shape[0], y + h + padding)

        cropped = gray[y_min:y_max, x_min:x_max]

        ch, cw = cropped.shape
        side = max(ch, cw)
        squared = np.zeros((side, side), dtype=np.uint8)
        offset_y = (side - ch) // 2
        offset_x = (side - cw) // 2
        squared[offset_y:offset_y+ch, offset_x:offset_x+cw] = cropped

        resized = cv2.resize(squared, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    def predict(self, canvas, top_k=3):
        """
        Returneaza top_k predictii ca lista de tuple (label, confidence).
        Daca canvas e gol, returneaza None.
        """
        processed = self.preprocess(canvas)
        if processed is None:
            return None

        predictions = self.model.predict(processed, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        return [(self.labels[i], float(predictions[i])) for i in top_indices]

    def predict_top1(self, canvas):
        """Returneaza doar (label, confidence) sau None."""
        result = self.predict(canvas, top_k=1)
        if result is None:
            return None
        return result[0]


if __name__ == "__main__":
    import sys

    classifier = DrawingClassifier()

    print("\nTest classifier - deseneaza ceva, apoi apasa SPACE pentru predictie")
    print("R = reset canvas | Q = quit\n")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    canvas = None
    prev_x, prev_y = 0, 0
    last_prediction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)
            dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
            if dist < 0.07:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
                if prev_x != 0:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (255, 255, 255), 4)
                prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = 0, 0

        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        if last_prediction:
            y_pos = 30
            for label, conf in last_prediction:
                color = (0, 255, 0) if conf > 0.5 else (0, 200, 200)
                cv2.putText(combined, f"{label}: {conf*100:.1f}%", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 30

        cv2.putText(combined, "SPACE = recunoaste | R = reset | Q = quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Test Classifier", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            last_prediction = None
            print("Canvas reset")
        elif key == ord(' '):
            preds = classifier.predict(canvas, top_k=3)
            if preds is None:
                print("Canvas gol!")
            else:
                last_prediction = preds
                print("\n--- Predictii ---")
                for label, conf in preds:
                    print(f"  {label}: {conf*100:.2f}%")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()