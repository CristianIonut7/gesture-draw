"""Script for training a CNN model on the Quick Draw dataset."""

# Set environment variables to reduce TensorFlow logging and disable oneDNN optimizations
import os # For setting environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import necessary libraries
import numpy as np # For numerical operations
from quickdraw import QuickDrawDataGroup # To access the Quick Draw dataset
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from PIL import Image  # For image processing
import tensorflow as tf # For building and training the CNN model
from tensorflow.keras.models import Sequential # For creating a sequential model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input # For defining layers in the CNN
from tensorflow.keras.utils import to_categorical # For converting labels to categorical format
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # For callbacks during training
import json # For saving label mappings

# Define constants and parameters for training
CATEGORIES = [
    "apple", "banana", "butterfly", "car", "cat", "circle", "cloud", "cup",
    "dog", "door", "eye", "fish", "flower", "hat", "house",
    "ice cream", "key", "leaf", "moon", "mountain", "mushroom",
    "smiley face", "snake", "star", "sun", "tree", "triangle", "umbrella",
    "candle", "donut"
]

# Training parameters
IMG_SIZE = 64 # Size to which images will be resized
SAMPLES_PER_CATEGORY = 2000 # Number of samples to use from each category
EPOCHS = 15 # Maximum number of epochs
BATCH_SIZE = 128 # Batch size
MODEL_DIR = "models" # Directory to save the trained model and label mappings
MODEL_PATH = os.path.join(MODEL_DIR, "quickdraw_model.h5") # Path to save the trained model
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json") # Path to save the label mappings

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Save label mappings to a JSON file for later use
with open(LABELS_PATH, "w") as f:
    json.dump(CATEGORIES, f)
print(f"Labels saved to {LABELS_PATH}")

print("\n--- Downloading and Preprocessing Quick Draw Images ---")
print(f"Categories: {len(CATEGORIES)} | Samples/Category: {SAMPLES_PER_CATEGORY}\n")

X = []
y = []

# Loop through each category, download the drawings, preprocess them, and store them in X and y
for label_idx, category in enumerate(CATEGORIES):
    print(f"  [{label_idx+1}/{len(CATEGORIES)}] {category}...")

    # Load the Quick Draw data for the current category
    qd_group = QuickDrawDataGroup(category, max_drawings=SAMPLES_PER_CATEGORY, recognized=True)

    # Loop through each drawing in the category, convert it to an image, preprocess it, and add it to the dataset
    count = 0
    for drawing in qd_group.drawings:
        img = drawing.get_image(stroke_color=(255, 255, 255), bg_color=(0, 0, 0), stroke_width=4) # Get the drawing as an image
        img = img.convert('L') # Convert the image to grayscale
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS) # Resize the image

        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
        X.append(arr)
        y.append(label_idx)
        count += 1
        if count >= SAMPLES_PER_CATEGORY:
            break

# Convert lists to numpy arrays and reshape X to have the correct dimensions for the CNN input
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
print(f"\nData shape: X={X.shape}, y={y.shape}")

# Split the dataset and convert the labels to categorical format
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes=len(CATEGORIES))
y_test_cat = to_categorical(y_test, num_classes=len(CATEGORIES))

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Build the CNN model architecture using Keras Sequential API, compile it with the Adam optimizer and categorical crossentropy loss, and print the model summary
print("\n--- Building CNN Model ---")
model = Sequential([
    # Define the input layer with the shape of the preprocessed images
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'), # First convolutional layer with 32 filters, 3x3 kernel size, ReLU activation, and same padding to preserve spatial dimensions
    Conv2D(32, (3, 3), activation='relu'), # Second convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    MaxPooling2D(pool_size=(2, 2)), # Max pooling layer to reduce spatial dimensions by taking the maximum value in 2x2 windows
    Dropout(0.25), # Dropout layer to reduce overfitting by randomly dropping 25% of the neurons during training

    # The second time we are doing this is to allow the model to learn more complex features
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # The third time we are doing this is to allow the model to learn even more complex features, especially since some categories may have subtle differences (e.g., "moon" vs "circle")
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(), # Flatten the output from the convolutional layers to feed into the fully connected layers
    Dense(256, activation='relu'), # Add a fully connected layer with ReLU activation to learn complex combinations of features
    Dropout(0.5), # Add dropout to reduce overfitting by randomly dropping some neurons during training
    Dense(len(CATEGORIES), activation='softmax') # Output layer with softmax activation to get probabilities for each category
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Use the Adam optimizer with a learning rate of 0.001
    loss='categorical_crossentropy', # Use categorical crossentropy loss for multi-class classification
    metrics=['accuracy'] # Track accuracy during training and evaluation
)
model.summary() # Print the model architecture summary

print("\n--- Starting Training ---")
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'), # Stop training if validation accuracy doesn't improve for 3 consecutive epochs and restore the best weights
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy') # Save the best model during training
]

history = model.fit(
    X_train, y_train_cat, # Training data and labels
    validation_data=(X_test, y_test_cat), # Validation data and labels
    epochs=EPOCHS, # Maximum number of epochs to train
    batch_size=BATCH_SIZE, # Batch size for training
    callbacks=callbacks, # Callbacks for early stopping and model checkpointing
    verbose=1 # Print progress during training
)

print("\n--- Final Evaluation ---")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"\nModel saved to: {MODEL_PATH}")