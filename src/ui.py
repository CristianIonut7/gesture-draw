"""UI rendering and interaction logic for the main menu."""

# Disable oneDNN optimizations and TensorFlow logging for cleaner output
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2 # OpenCV for rendering and UI interaction
import numpy as np # NumPy for image manipulation and effects

# Constants for display dimensions, colors, and menu options
DISPLAY_W, DISPLAY_H = 1280, 720
HOVER_SPEED = 0.08

COLOR_PAPER = (220, 215, 200)
COLOR_PAPER_DARK = (195, 188, 170)
COLOR_INK = (45, 40, 50)
COLOR_INK_LIGHT = (95, 85, 95)
COLOR_RED = (60, 70, 165)
COLOR_BLUE = (130, 90, 70)
COLOR_OLIVE = (75, 110, 130)

MENU_OPTIONS = [
    {"id": "free", "title": "Free Draw",      "subtitle": "draw whatever you want",       "accent": COLOR_OLIVE},
    {"id": "solo", "title": "Solo Challenge", "subtitle": "let the AI guess your drawing","accent": COLOR_RED},
    {"id": "duel", "title": "1v1 Battle",     "subtitle": "challenge a friend",           "accent": COLOR_BLUE},
    {"id": "quit", "title": "Quit",           "subtitle": "exit application",             "accent": COLOR_INK_LIGHT},
]

# Create a textured paper-like background
def make_paper_background(w, h):
    bg = np.full((h, w, 3), COLOR_PAPER, dtype=np.uint8)
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - 0.15 * (dist / max_dist)
    bg = (bg * vignette[..., np.newaxis]).astype(np.uint8)
    return bg

# Fade out effect
def fade_out(window_name, current_frame, frames=12): 
    for i in range(frames):
        alpha = 1 - (i / frames)
        faded = (current_frame * alpha).astype(np.uint8)
        cv2.imshow(window_name, faded)
        cv2.waitKey(1)

# Fade in effect
def fade_in(window_name, target_frame, frames=12):
    for i in range(frames + 1):
        alpha = i / frames
        faded = (target_frame * alpha).astype(np.uint8)
        cv2.imshow(window_name, faded)
        cv2.waitKey(1)

# Easing function for smooth hover animations
def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

# Draw a button on the frame
def draw_button(frame, x, y, w, h, option, hover_progress=0.0):
    accent = option["accent"] # Use the option's accent color for the button
    eased = ease_out_cubic(hover_progress) # Apply easing to the hover progress for smoother animation

    
    if hover_progress > 0: # Only draw the hover effect if there's some hover progress
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), accent, -1)
        cv2.addWeighted(overlay, 0.18 * eased, frame, 1 - 0.18 * eased, 0, frame)

    # Draw the progress line
    line_start_x = x + 60
    max_line_w = w - 60 - 30
    line_w = int(max_line_w * (0.6 + 0.4 * eased))
    cv2.line(frame, (line_start_x, y + h - 8), (line_start_x + line_w, y + h - 8),
             accent, 2, cv2.LINE_AA)

    # Draw the circular marker
    marker_x = x + 30
    marker_y = y + h // 2
    if hover_progress > 0.05:
        inner_radius = max(1, int(5 * eased))
        cv2.circle(frame, (marker_x, marker_y), 5, accent, 1, cv2.LINE_AA)
        cv2.circle(frame, (marker_x, marker_y), inner_radius, accent, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (marker_x, marker_y), 5, accent, 1, cv2.LINE_AA)

    # Draw the title and subtitle text
    title_offset = int(8 * eased)
    cv2.putText(frame, option["title"], (x + 60 + title_offset, y + h // 2 - 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, COLOR_INK, 1, cv2.LINE_AA)
    cv2.putText(frame, option["subtitle"], (x + 60 + title_offset, y + h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

# get the layout for the menu buttons
def get_button_layout():
    btn_w = 600
    btn_h = 80
    btn_spacing = 18
    total_h = len(MENU_OPTIONS) * btn_h + (len(MENU_OPTIONS) - 1) * btn_spacing
    start_y = (DISPLAY_H - total_h) // 2 + 60
    start_x = (DISPLAY_W - btn_w) // 2
    return [(opt, start_x, start_y + i * (btn_h + btn_spacing), btn_w, btn_h)
            for i, opt in enumerate(MENU_OPTIONS)]

# Check if a point is within a rectangle
def point_in_rect(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h

# Global state for mouse position and click status
_mouse_state = {"x": -1, "y": -1, "clicked": False}

# Mouse callback to update the mouse state
def _mouse_callback(event, x, y, flags, param):
    _mouse_state["x"] = x
    _mouse_state["y"] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_state["clicked"] = True

# Cache the paper background to avoid regenerating it every frame
_cached_paper_bg = None

# Render the main menu frame
def render_menu_frame(hover_state):
    global _cached_paper_bg
    if _cached_paper_bg is None:
        _cached_paper_bg = make_paper_background(DISPLAY_W, DISPLAY_H)
    display = _cached_paper_bg.copy()

    # Draw the title and decorative lines
    title = "GestureWar"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 2)[0]
    cv2.putText(display, title, ((DISPLAY_W - title_size[0]) // 2, 110),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_INK, 2, cv2.LINE_AA)

    # Draw decorative lines around the title
    line_y = 135
    cv2.line(display, (DISPLAY_W // 2 - 200, line_y),
             (DISPLAY_W // 2 - 30, line_y), COLOR_INK_LIGHT, 1, cv2.LINE_AA)
    cv2.line(display, (DISPLAY_W // 2 + 30, line_y),
             (DISPLAY_W // 2 + 200, line_y), COLOR_INK_LIGHT, 1, cv2.LINE_AA)
    cv2.circle(display, (DISPLAY_W // 2, line_y), 4, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

    # Draw the subtitle
    subtitle = "draw with your hand"
    sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
    cv2.putText(display, subtitle, ((DISPLAY_W - sub_size[0]) // 2, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

    # Draw the menu buttons
    layout = get_button_layout()
    for option, bx, by, bw, bh in layout:
        draw_button(display, bx, by, bw, bh, option, hover_state[option["id"]])

    # Draw the footer text
    footer = "click an option to begin"
    f_size = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(display, footer, ((DISPLAY_W - f_size[0]) // 2, DISPLAY_H - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

    return display


def run_menu(window_name="GestureWar", fade_in_first=False):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_callback)

    hover_state = {opt["id"]: 0.0 for opt in MENU_OPTIONS}
    selected_option = None
    _mouse_state["clicked"] = False
    display = None

    if fade_in_first:
        first_frame = render_menu_frame(hover_state)
        fade_in(window_name, first_frame)

    # Main loop to handle menu interaction
    while selected_option is None:
        # Update hover states based on mouse position
        for option_id in hover_state:
            opt_data = next(o for o in MENU_OPTIONS if o["id"] == option_id)
            layout = get_button_layout()
            bx, by, bw, bh = next((x, y, w, h) for o, x, y, w, h in layout if o["id"] == option_id)
            is_hovered = point_in_rect(_mouse_state["x"], _mouse_state["y"], bx, by, bw, bh)
            target = 1.0 if is_hovered else 0.0
            current = hover_state[option_id]
            hover_state[option_id] = current + (target - current) * HOVER_SPEED

        # Render the menu frame with updated hover states
        display = render_menu_frame(hover_state)

        # Check for clicks on buttons
        layout = get_button_layout()
        for option, bx, by, bw, bh in layout:
            is_hovered = point_in_rect(_mouse_state["x"], _mouse_state["y"], bx, by, bw, bh)
            if _mouse_state["clicked"] and is_hovered:
                selected_option = option["id"]
        _mouse_state["clicked"] = False
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            selected_option = "quit"
            break

    return selected_option, display


if __name__ == "__main__":
    choice, _ = run_menu()
    print(f"Selected: {choice}")
    cv2.destroyAllWindows()