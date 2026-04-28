# Training the CNN Model

This document explains how to train the drawing classifier from scratch, what parameters can be tuned, and how to add new categories.

For the inference side (using a trained model), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick Start

```bash
python train_model.py
```

This downloads ~60 MB of drawings from the Quick Draw API, trains a CNN for up to 15 epochs, and saves the result to `models/quickdraw_model.h5`.

**Expected runtime:** 45-90 minutes on CPU, 5-15 minutes on GPU.

---

## What `train_model.py` Does

The script is split into four phases:

### 1. Data download

For each of the 30 categories defined in `CATEGORIES`, the [`quickdraw`](https://pypi.org/project/quickdraw/) Python package downloads up to `SAMPLES_PER_CATEGORY` user-recognized drawings from Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset).

Each drawing is stored as a sequence of strokes (vector data); the script renders each one as a 64×64 grayscale PNG with white strokes on black background, matching the look of in-game drawings.

The downloaded data is cached in `~/.quickdrawcache` so subsequent runs skip the download.

### 2. Train/test split

The 60,000 images are split using `sklearn.train_test_split` with `stratify=y` to preserve per-category proportions:
- **Training set:** 51,000 images (85%)
- **Test set:** 9,000 images (15%)

### 3. Model construction

A VGG-style CNN with three convolutional blocks and a dense classification head:

```
Input(64×64×1)
├── Conv2D(32, 3×3, padding=same, ReLU)
├── Conv2D(32, 3×3, ReLU)
├── MaxPool(2×2)
├── Dropout(0.25)
├── Conv2D(64, 3×3, padding=same, ReLU)
├── Conv2D(64, 3×3, ReLU)
├── MaxPool(2×2)
├── Dropout(0.25)
├── Conv2D(128, 3×3, padding=same, ReLU)
├── MaxPool(2×2)
├── Dropout(0.25)
├── Flatten
├── Dense(256, ReLU)
├── Dropout(0.5)
└── Dense(30, softmax)
```

Total parameters: ~600,000.

### 4. Training

- **Optimizer:** Adam with learning rate 0.001
- **Loss:** categorical crossentropy
- **Batch size:** 128
- **Max epochs:** 15
- **EarlyStopping** monitoring `val_accuracy` with patience=3 - training stops if no improvement for 3 consecutive epochs
- **ModelCheckpoint** saves only the best version of the model based on `val_accuracy`

The final model achieves around **92.6% accuracy** on the test set.

---

## Tunable Parameters

All parameters live at the top of `train_model.py`:

| Parameter | Default | Effect |
|---|---|---|
| `IMG_SIZE` | 64 | Input image size. Larger = more detail but slower training and more memory. 64 is a good balance. |
| `SAMPLES_PER_CATEGORY` | 2000 | Drawings per category. More data = better accuracy but longer training time. |
| `EPOCHS` | 15 | Maximum training cycles. EarlyStopping will end sooner if the model converges. |
| `BATCH_SIZE` | 128 | How many images are processed at once. Larger batches train faster but use more memory. |

### Tradeoffs

**Want better accuracy?**
- Increase `SAMPLES_PER_CATEGORY` to 5000 or 10000 (training time scales linearly)
- Increase `EPOCHS` to 25 (EarlyStopping will still cut off if no improvement)

**Want faster training?**
- Decrease `SAMPLES_PER_CATEGORY` to 500-1000 (accuracy drops to ~85%)
- Decrease `IMG_SIZE` to 32 (much faster, but loses detail)

**GPU available?**
- Make sure TensorFlow detects it: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- If empty, install GPU-enabled TensorFlow: see [TF GPU setup guide](https://www.tensorflow.org/install/gpu)
- Increase `BATCH_SIZE` to 256 or 512 to use the GPU more efficiently

---

## Adding New Categories

The Quick Draw dataset has **345 categories**, far more than the 30 we use. Adding new ones is easy:

1. **Pick categories** — see the [full list](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt). Choose drawings that:
   - Are visually distinct from existing categories (avoid `dog` if you have `cat`)
   - Are simple enough to draw with a finger in mid-air (avoid `The Mona Lisa`)
   - Have a clear shape (avoid abstract concepts like `nothing`)

2. **Edit `train_model.py`:**
   ```python
   CATEGORIES = [
       "apple", "banana", # ... existing
       "rocket", "guitar", "fork",  # new ones
   ]
   ```

3. **Edit `assets/words.json`** to add the same new words (so they can be picked as game targets).

4. **Re-train** — `python train_model.py`. The first run will download the new categories' data.

5. **Optional: clean up `~/.quickdrawcache`** if it gets too big.

### Notes on category count

- More categories = harder classification problem. Going from 30 to 100 categories typically drops accuracy by 5-10 percentage points, all else equal.
- Some categories overlap heavily (`cat`/`dog`, `mountain`/`triangle`). Adding both means more confusions in practice.
- The output layer must match: `Dense(len(CATEGORIES), activation='softmax')` - already wired up correctly.

---

## Inspecting the Trained Model

After training, you can examine the model's behavior:

```python
import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model("models/quickdraw_model.h5")
labels = json.load(open("models/labels.json"))

# Show architecture
model.summary()

# Total parameter count
total = model.count_params()
print(f"Total params: {total:,}")
```

To compute a confusion matrix on the test set:

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# (assuming X_test, y_test are available — re-run the data prep section of train_model.py)
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.savefig('docs/confusion_matrix.png', dpi=150)
```

This reveals which categories the model confuses most often - useful for deciding which to remove or retrain with more data.

---

## Common Issues

### Out of memory during training

Reduce `BATCH_SIZE` to 64 or 32. The model weights are small (~7 MB), but the per-batch gradient computations need space.

### Quick Draw download is very slow

The package fetches data sequentially per category. There's no parallelism built in. For 30 categories × 2000 samples, expect 5-15 minutes for the download phase.

If you have intermittent network issues, the cache lets you re-run; only missing categories will re-download.

### Final accuracy is much lower than 92%

Sanity checks:
- `SAMPLES_PER_CATEGORY` should be ≥ 1000 — fewer than that and the model overfits
- Make sure `recognized=True` in the `QuickDrawDataGroup` call — this filters out drawings the original game didn't recognize, which are usually low-quality
- Make sure your `CATEGORIES` list doesn't have typos — invalid category names just return 0 drawings silently

### EarlyStopping triggers too soon

If training stops at epoch 5-6 with mediocre accuracy, the model probably needs more capacity:
- Try increasing the dense layer width: `Dense(512, activation='relu')` instead of 256
- Try adding another conv block before the dense layer
- Reduce dropout values (e.g., 0.2 instead of 0.25)

But also: 92% with EarlyStopping at epoch 12 is normal and fine. Don't over-engineer.

---

## Reproducibility

To get bit-for-bit reproducible results:
```python
import tensorflow as tf
import numpy as np
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

Add this at the top of `train_model.py`. Note that even with seeds, slight variations can occur due to GPU non-determinism and TensorFlow version differences.
