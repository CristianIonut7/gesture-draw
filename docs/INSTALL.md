# Installation Guide

This guide covers a more detailed installation process than the README quickstart, including troubleshooting for common issues.

---

## System Requirements

- **OS:** Windows 10/11 (primary target). Linux and macOS should work but are untested.
- **Python:** 3.11.x (3.10 also works; avoid 3.12+ as TensorFlow support varies)
- **Memory:** 4 GB RAM minimum, 8 GB recommended
- **Camera:** any USB or built-in webcam at 640×480 resolution or higher
- **Disk:** ~2 GB total (TensorFlow + dataset cache + model files)
- **GPU:** not required; the CNN runs fast enough on CPU

---

## Step-by-step on Windows

### 1. Install Python 3.11

Recommended way: use [pyenv-win](https://github.com/pyenv-win/pyenv-win) to manage Python versions cleanly.

```powershell
# Install pyenv-win via PowerShell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
./install-pyenv-win.ps1

# Restart PowerShell, then install Python 3.11.9
pyenv install 3.11.9
pyenv shell 3.11.9
```

Alternative: download from [python.org](https://www.python.org/downloads/) and install - make sure to check "Add Python to PATH" during installation.

Verify the install:
```powershell
python --version
# Should print: Python 3.11.x
```

### 2. Clone the repository

```powershell
git clone https://github.com/<your-username>/gesture-draw.git
cd gesture-draw
```

### 3. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this once (admin shell):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Your prompt should now show `(venv)` at the start.

### 4. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `tensorflow` (~500 MB download)
- `opencv-python`
- `mediapipe`
- `numpy`
- `Pillow`
- `scikit-learn`
- `quickdraw` (only needed if you train the model yourself)

The first install takes 5-10 minutes depending on your internet speed.

### 5. Get the trained CNN model

You have two options:

**Option A - train it yourself (recommended for first-time users):**
```powershell
python train_model.py
```
This downloads ~60 MB of drawings from the Quick Draw API (cached in `~/.quickdrawcache`) and trains the CNN. Takes ~45-90 minutes on CPU.

**Option B - download pre-trained model:**
> If a pre-trained model is published in a [Release](../../releases), download `quickdraw_model.h5` and place it in `models/`.

### 6. Run the application

```powershell
python main.py
```

The MediaPipe hand-tracking model (~25 MB) downloads automatically on first run. The main window should appear with the GestureWar menu after 5-10 seconds.

---

## Troubleshooting

### "No module named 'cv2'" / "No module named 'tensorflow'"

You forgot to activate the virtual environment. Run:
```powershell
.\venv\Scripts\Activate.ps1
```

### Camera doesn't open / black screen

- **Check the camera works** in another app (Camera app on Windows)
- **Close other apps** that might be using the camera (Zoom, Teams, OBS)
- **Try a different camera index** — edit `canvas.py` and change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras

### TensorFlow GPU warnings on startup

You'll see messages like *"Could not load dynamic library 'cudart64_110.dll'"*. This is **normal and harmless** - TensorFlow tried to find a GPU and falls back to CPU automatically. The app runs fine on CPU.

To suppress these warnings, they're already disabled by:
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

### "Permission denied" or virtualenv issues on PowerShell

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Application is laggy (low FPS)

- **Close background apps** that use CPU (Chrome, Discord, etc.)
- **Plug in your laptop** - Windows reduces CPU frequency on battery, dropping FPS by ~25%
- **Check the [performance notes](ARCHITECTURE.md#performance)** for expected FPS on different hardware

### "Module not found: quickdraw"

`quickdraw` is only needed for training. If you didn't install it and only want to run the game (with a pre-trained model), this is not an issue. To install it manually:
```powershell
pip install quickdraw
```

### Python 3.12+ compatibility

TensorFlow 2.x has limited support for Python 3.12+. Stick to Python 3.10 or 3.11 to avoid installation issues.

### "WSL camera access" issues

If you're on WSL (Windows Subsystem for Linux), camera access requires extra setup and is generally fragile. **Strongly recommended:** run on native Windows instead.

---

## Updating

To update an existing installation:
```powershell
git pull
.\venv\Scripts\Activate.ps1
pip install --upgrade -r requirements.txt
```

---

## Uninstall

To completely remove the project:
```powershell
# Deactivate venv if active
deactivate

# Remove the project folder
cd ..
Remove-Item -Recurse -Force gesture-draw

# Optionally clear Quick Draw cache (if you trained the model)
Remove-Item -Recurse -Force "$HOME\.quickdrawcache"
```

The MediaPipe model is downloaded inside the project folder, so removing the project also removes it.
