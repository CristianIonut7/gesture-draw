"""
modes.py
Game modes - functions called by main.py based on menu selection.
Each mode receives a shared GestureCanvas and returns the last frame
(used for fade-out transition back to menu).
"""
import cv2


def run_free_draw(canvas, window_name="GestureWar"):
    """
    Free Draw mode - draw whatever you want.
    ESC or Q returns to menu.
    Returns the last displayed frame.
    """
    canvas.reset_canvas()
    last_display = None

    while True:
        state = canvas.update()
        if state is None:
            break

        display = state["display"]
        canvas.draw_color_palette(display)

        cv2.putText(display, "Free Draw - draw whatever you want",
                    (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)

        if state["gesture_label"]:
            cv2.putText(display, state["gesture_label"],
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        state["current_color"], 2, cv2.LINE_AA)

        h = display.shape[0]
        cv2.putText(display, "ESC = back to menu",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    return last_display


def run_solo_challenge(canvas, window_name="GestureWar"):
    """Placeholder."""
    print("Solo Challenge - coming next!")
    return None


def run_duel(canvas, window_name="GestureWar"):
    """Placeholder."""
    print("1v1 Battle - coming after solo!")
    return None