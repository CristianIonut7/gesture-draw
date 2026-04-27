import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from quickdraw import QuickDrawDataGroup
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json


CATEGORIES = [
    "apple", "banana", "butterfly", "car", "cat", "circle", "cloud", "cup",
    "dog", "door", "eye", "fish", "flower", "hat", "house",
    "ice cream", "key", "leaf", "moon", "mountain", "mushroom",
    "smiley face", "snake", "star", "sun", "tree", "triangle", "umbrella",
    "candle", "donut"
]

IMG_SIZE = 64
SAMPLES_PER_CATEGORY = 2000
EPOCHS = 15
BATCH_SIZE = 128
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "quickdraw_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(LABELS_PATH, "w") as f:
    json.dump(CATEGORIES, f)
print(f"Etichete salvate in {LABELS_PATH}")

print("\n--- Descarcare si preprocesare desene Quick Draw ---")
print(f"Categorii: {len(CATEGORIES)} | Sample-uri/categorie: {SAMPLES_PER_CATEGORY}\n")

X = []
y = []

for label_idx, category in enumerate(CATEGORIES):
    print(f"  [{label_idx+1}/{len(CATEGORIES)}] {category}...")

    qd_group = QuickDrawDataGroup(category, max_drawings=SAMPLES_PER_CATEGORY, recognized=True)

    count = 0
    for drawing in qd_group.drawings:
        img = drawing.get_image(stroke_color=(255, 255, 255), bg_color=(0, 0, 0), stroke_width=4)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        arr = np.array(img, dtype=np.float32) / 255.0
        X.append(arr)
        y.append(label_idx)
        count += 1
        if count >= SAMPLES_PER_CATEGORY:
            break

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
print(f"\nForma date: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes=len(CATEGORIES))
y_test_cat = to_categorical(y_test, num_classes=len(CATEGORIES))

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

print("\n--- Construire model CNN ---")
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CATEGORIES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\n--- Incep antrenarea ---")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n--- Evaluare finala ---")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Acuratete pe test: {test_acc*100:.2f}%")
print(f"\nModel salvat in: {MODEL_PATH}")