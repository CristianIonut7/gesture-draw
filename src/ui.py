"""
ui.py
Main menu in vintage / paper minimalist style.
Mouse selection only.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

DISPLAY_W, DISPLAY_H = 1280, 720
HOVER_SPEED = 0.05 

COLOR_PAPER = (220, 215, 200)
COLOR_PAPER_DARK = (195, 188, 170)
COLOR_INK = (45, 40, 50)
COLOR_INK_LIGHT = (95, 85, 95)
COLOR_RED = (60, 70, 165)
COLOR_BLUE = (130, 90, 70)
COLOR_OLIVE = (75, 110, 130)

MENU_OPTIONS = [
    {"id": "free", "title": "Free Draw",      "subtitle": "draw whatever you want",      "accent": COLOR_OLIVE},
    {"id": "solo", "title": "Solo Challenge", "subtitle": "let the AI guess your drawing","accent": COLOR_RED},
    {"id": "duel", "title": "1v1 Battle",     "subtitle": "challenge a friend",           "accent": COLOR_BLUE},
    {"id": "quit", "title": "Quit",           "subtitle": "exit application",             "accent": COLOR_INK_LIGHT},
]


def make_paper_background(w, h):
    """Paper background with subtle texture and vignette."""
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


def ease_out_cubic(t):
    """Smooth easing function for animations."""
    return 1 - (1 - t) ** 3


def draw_button(frame, x, y, w, h, option, hover_progress=0.0):
    """Vintage style button with smooth hover animation."""
    accent = option["accent"]
    eased = ease_out_cubic(hover_progress)

    if hover_progress > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), accent, -1)
        cv2.addWeighted(overlay, 0.18 * eased, frame, 1 - 0.18 * eased, 0, frame)

    line_start_x = x + 60 
    max_line_w = w - 60 - 30 
    line_w = int(max_line_w * (0.55 + 0.45 * eased))
    cv2.line(frame, (line_start_x, y + h - 8), (line_start_x + line_w, y + h - 8),
             accent, 2, cv2.LINE_AA)
    marker_x = x + 30
    marker_y = y + h // 2
    if hover_progress > 0.05:
        inner_radius = max(1, int(5 * eased))
        cv2.circle(frame, (marker_x, marker_y), 5, accent, 1, cv2.LINE_AA)
        cv2.circle(frame, (marker_x, marker_y), inner_radius, accent, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (marker_x, marker_y), 5, accent, 1, cv2.LINE_AA)

    title_offset = int(8 * eased)
    cv2.putText(frame, option["title"], (x + 60 + title_offset, y + h // 2 - 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, COLOR_INK, 1, cv2.LINE_AA)

    cv2.putText(frame, option["subtitle"], (x + 60 + title_offset, y + h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_INK_LIGHT, 1, cv2.LINE_AA)


def get_button_layout():
    btn_w = 600
    btn_h = 80
    btn_spacing = 18
    total_h = len(MENU_OPTIONS) * btn_h + (len(MENU_OPTIONS) - 1) * btn_spacing
    start_y = (DISPLAY_H - total_h) // 2 + 60
    start_x = (DISPLAY_W - btn_w) // 2
    return [(opt, start_x, start_y + i * (btn_h + btn_spacing), btn_w, btn_h)
            for i, opt in enumerate(MENU_OPTIONS)]


def point_in_rect(px, py, x, y, w, h):
    return x <= px <= x + w and y <= py <= y + h


mouse_x, mouse_y = -1, -1
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


def run_menu():
    global mouse_clicked

    cv2.namedWindow("GestureWar", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("GestureWar", mouse_callback)

    paper_bg = make_paper_background(DISPLAY_W, DISPLAY_H)

    hover_state = {opt["id"]: 0.0 for opt in MENU_OPTIONS}
    selected_option = None
    import time
    last_time = time.time()
    fps_counter = 0
    fps_display = 0

    while selected_option is None:
        display = paper_bg.copy()
        

        title = "GestureWar"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 2)[0]
        title_x = (DISPLAY_W - title_size[0]) // 2
        cv2.putText(display, title, (title_x, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_INK, 2, cv2.LINE_AA)

        line_y = 135
        cv2.line(display, (DISPLAY_W // 2 - 200, line_y),
                 (DISPLAY_W // 2 - 30, line_y), COLOR_INK_LIGHT, 1, cv2.LINE_AA)
        cv2.line(display, (DISPLAY_W // 2 + 30, line_y),
                 (DISPLAY_W // 2 + 200, line_y), COLOR_INK_LIGHT, 1, cv2.LINE_AA)
        cv2.circle(display, (DISPLAY_W // 2, line_y), 4, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

        subtitle = "draw with your hand"
        sub_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(display, subtitle,
                    ((DISPLAY_W - sub_size[0]) // 2, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

        layout = get_button_layout()
        for option, bx, by, bw, bh in layout:
            is_hovered = point_in_rect(mouse_x, mouse_y, bx, by, bw, bh)

            target = 1.0 if is_hovered else 0.0
            current = hover_state[option["id"]]
            hover_state[option["id"]] = current + (target - current) * HOVER_SPEED

            draw_button(display, bx, by, bw, bh, option, hover_state[option["id"]])

            if mouse_clicked and is_hovered:
                selected_option = option["id"]
        mouse_clicked = False

        footer = "click an option to begin"
        f_size = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(display, footer,
                    ((DISPLAY_W - f_size[0]) // 2, DISPLAY_H - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INK_LIGHT, 1, cv2.LINE_AA)

        fps_counter += 1
        if time.time() - last_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            last_time = time.time()
        cv2.putText(display, f"FPS: {fps_display}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INK_LIGHT, 1, cv2.LINE_AA)
        cv2.imshow("GestureWar", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            selected_option = "quit"
            break

    return selected_option


if __name__ == "__main__":
    choice = run_menu()
    print(f"Selected: {choice}")
    cv2.destroyAllWindows()