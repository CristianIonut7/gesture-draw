# Gesture Cheat Sheet

GestureWar uses four hand gestures detected in real time by MediaPipe. This document explains how to perform each one, what it does in different game modes, and what to do if a gesture isn't being detected reliably.

---

## The Four Gestures

| Gesture | Visual | Detection rule (technical) | Free Draw | Solo | Duel |
|:---:|:---|:---|:---:|:---:|:---:|
| ☝️ **Index up** | One finger pointing | Default state — index visible, no other gesture | ✏️ Draw | ✏️ Draw | ✏️ Draw |
| 🤏 **Pinch** | Thumb + index touching | Thumb-index distance < 0.07 (normalized) | 🛑 Pen up | 🛑 Pen up | 🛑 Pen up |
| 🖐️ **Open hand** | All fingers spread | All fingertips above PIP joints + thumb-index distance > 0.1 | 🎨 Change color | 🤖 Call AI | — |
| ✊ **Fist** | All fingers curled | All fingertips below MCP joints | 🧹 Clear canvas | 🧹 Clear canvas | 🧹 Clear your half |

---

## How to Perform Each Gesture

### ☝️ Index up (drawing)

This is the default state when nothing else is detected. Keep your index finger extended and held visible to the camera. The other fingers can be loosely curled or extended - what matters is that the **pinch distance** stays large (you're not grabbing).

**Tip:** Move slowly at first. The camera captures at ~30 FPS, so very fast hand motion can cause gaps in your drawing.

### 🤏 Pinch (pen up)

Bring your thumb tip and index fingertip together. They don't need to actually touch - just close enough that the normalized distance is small.

This is the gesture you use **between strokes** when you don't want to leave a trail. Otherwise, every motion of your hand would draw a line.

**Tip:** Practice picking up your "pen" smoothly. A common beginner mistake is to draw, then drop your hand to your side without pinching first - leaving an unwanted line across the canvas.

### 🖐️ Open hand (~0.5s hold)

Open your palm fully toward the camera with all fingers spread. Hold the position for about half a second. A progress bar appears on screen showing how long you have to hold.

This is a **deliberate action gesture** - accidental brief opening of the hand won't trigger it because of the half-second hold requirement.

**Used for:**
- **Free Draw:** cycle to the next color
- **Solo Challenge:** call the AI to evaluate your drawing
- **Duel:** *not used* (AI runs automatically every 1.2s)

### ✊ Fist (~0.5s hold)

Curl all four fingers down so the fingertips are below the MCP joints (the knuckles closest to the palm). Hold for about half a second; a red progress bar will appear.

This is a destructive action - it clears your canvas - so the half-second hold prevents accidental erasure.

**In 1v1 mode** the fist only clears **your own half** of the canvas, not your opponent's.

---

## When Gestures Aren't Working

MediaPipe is generally robust but sometimes struggles. Here are the most common issues:

### Drawing keeps cutting out

The pinch detection is too sensitive - your thumb is wandering close to your index even when you're trying to draw.

**Fix:** keep your thumb tucked toward your palm or out to the side while drawing. Only bring it near the index when you specifically want to pen-up.

### Open hand triggers when I just want to draw

Your fingers are all extended visibly when you draw normally. The detector sees this as an open hand.

**Fix:** keep your middle, ring, and pinky fingers slightly curled while drawing. Only spread all five fully when you want the open-hand action.

### Fist isn't detected reliably

The detection rule requires all four fingertips to be **below** their MCP joints on the Y axis. If your hand is tilted (e.g., side-on to the camera), this fails.

**Fix:** show your fist with the back of your hand directly facing the camera, fingers curled inward.

### Hand isn't detected at all

MediaPipe needs:
- **Good lighting** - even indirect daylight is better than dim indoor lighting
- **Hand fully in the frame** - don't let your fingers go off-screen
- **Some skin contrast** with the background - a complex background is fine, but the hand should be visually distinguishable

If detection is still flaky, try moving 30-50 cm from the camera (closer than typical webcam distance).

### Wrong player attribution in 1v1

In 1v1 mode, hands are assigned to players based on horizontal position: hands on the left half are Player 1, right half are Player 2. If hands cross over, they may swap.

**Fix:** stay on your half of the camera frame, and if you reach toward the center, do so deliberately so MediaPipe can re-assign.

---

## Tips for Better Drawings

### Draw bigger

The CNN was trained on 64×64 pixel images. Tiny details (sub-pixel features) get lost in the resize. Drawing a star that fills 30% of the canvas works much better than one that fills 5%.

### Use distinct shapes

The Quick Draw dataset has some categories that are visually similar and easily confused:
- `cat` ↔ `dog`
- `circle` ↔ `moon` ↔ `sun`
- `mountain` ↔ `triangle`
- `key` ↔ `pencil`

If you have to draw one of these, make the distinctive features obvious (whiskers for cat, rays for sun, etc.).

### Don't worry about quality

The CNN was trained on rapid sketches by random users - many of which are pretty bad. It's tuned for "the gist of the shape," not artistic quality. A wobbly-but-recognizable star scores better than a perfect one drawn very slowly.

### Don't redraw the same thing 5 times

In Solo Challenge, if the AI guesses wrong, the wrong word goes on a "banned" list - the next call will skip it. So if you've drawn a clear star and the AI keeps saying "starfish," try opening your hand again immediately - its second guess is more likely to be correct.

---

## Adding New Gestures

If you want to extend GestureWar with new gestures, the place to add them is in `src/canvas.py`. Look at how `is_fist()` and `is_open_hand()` are implemented - they're simple geometric checks over the 21 landmarks.

A few ideas for new gestures:
- **Three fingers up** (index + middle + ring): undo last stroke
- **L-shape** (thumb + index, others curled): switch tool (e.g., between brush and eraser)
- **Two fingers** (index + middle, others curled): toggle a guide grid

Adding a gesture means:
1. Write a `is_my_gesture(landmarks)` function in `canvas.py`
2. Hook it into `GestureCanvas.update()` between the existing checks
3. Decide what action it triggers and whether it should require a hold (use a frame counter like `fist_frames`)
