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
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)

    def preprocess(self, canvas):
        if len(canvas.shape) == 3:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:
            gray = canvas.copy()

        if cv2.countNonZero(gray) < 50:
            return None

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
        processed = self.preprocess(canvas)
        if processed is None:
            return None

        predictions = self.model.predict(processed, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        return [(self.labels[i], float(predictions[i])) for i in top_indices]

    def predict_top1(self, canvas):
        result = self.predict(canvas, top_k=1)
        if result is None:
            return None
        return result[0]