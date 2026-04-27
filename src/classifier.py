""" A class to classify hand-drawn sketches using a pre-trained TensorFlow model."""

# Suppress TensorFlow logging and warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimizations to avoid potential issues on some systems
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all TensorFlow logs

# Import necessary libraries
import numpy as np # For numerical operations
import cv2  # For image processing
import json # For loading label mappings
import tensorflow as tf # For loading and using the pre-trained model


MODEL_PATH = "models/quickdraw_model.h5"
LABELS_PATH = "models/labels.json"
IMG_SIZE = 64

# The DrawingClassifier class encapsulates the functionality to classify hand-drawn sketches.
class DrawingClassifier:
    # Constructor to load the model and labels
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
    
    # Preprocess the input canvas to prepare it for prediction
    def preprocess(self, canvas):
        if len(canvas.shape) == 3: # If the canvas is in color, convert it to grayscale
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        else:
            gray = canvas.copy()

        # If the canvas is mostly empty, return None to indicate no prediction
        if cv2.countNonZero(gray) < 50:
            return None

        
        coords = cv2.findNonZero(gray) # Find the coordinates of the drawing
        x, y, w, h = cv2.boundingRect(coords) # Get the bounding box

        # Add padding around the bounding box
        padding = 20
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(gray.shape[1], x + w + padding)
        y_max = min(gray.shape[0], y + h + padding)

        # Crop the image to the bounding box
        cropped = gray[y_min:y_max, x_min:x_max]

        # Resize the cropped image to a square and maintain aspect ratio
        ch, cw = cropped.shape
        side = max(ch, cw)
        squared = np.zeros((side, side), dtype=np.uint8)
        offset_y = (side - ch) // 2
        offset_x = (side - cw) // 2
        squared[offset_y:offset_y+ch, offset_x:offset_x+cw] = cropped

        # Resize the squared image
        resized = cv2.resize(squared, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Normalize pixel values and reshape for model input
        normalized = resized.astype(np.float32) / 255.0
        return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict the class of the drawing
    def predict(self, canvas, top_k=3):
        processed = self.preprocess(canvas)
        if processed is None:
            return None
        
        predictions = self.model.predict(processed, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        return [(self.labels[i], float(predictions[i])) for i in top_indices]

    # Predict the top class for the drawing
    def predict_top1(self, canvas):
        result = self.predict(canvas, top_k=1)
        if result is None:
            return None
        return result[0]