"""Main entry point for the GestureWar game. Handles the main loop, menu navigation, and mode launching."""

#Add folder src to path so we can import from it
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

#Import OpenCV and UI components
import cv2
from ui import (
    run_menu, fade_in, fade_out,
    make_paper_background,
    COLOR_INK, COLOR_INK_LIGHT,
    DISPLAY_W, DISPLAY_H,
)

# Define the window name for OpenCV display
WINDOW_NAME = "GestureWar"

# Function to show a loading screen while initializing the camera and canvas
def show_loading_screen(message="Loading..."):
    bg = make_paper_background(DISPLAY_W, DISPLAY_H)
    title = "GestureWar"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 2)[0]
    cv2.putText(bg, title, ((DISPLAY_W - title_size[0]) // 2, DISPLAY_H // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_INK, 2, cv2.LINE_AA)
    msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
    cv2.putText(bg, message, ((DISPLAY_W - msg_size[0]) // 2, DISPLAY_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INK_LIGHT, 1, cv2.LINE_AA)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW_NAME, bg)
    cv2.waitKey(1)
    return bg # Return the background image so we can use it for fading effects

def main():
    shared_canvas = None   # This will hold the GestureCanvas instance that can be shared across modes to preserve drawings
    fade_in_menu_next = False

    # Function to initialize and return the shared GestureCanvas instance. It will create the canvas if it doesn't exist yet
    def get_canvas():
        nonlocal shared_canvas
        if shared_canvas is None:
            show_loading_screen("warming up the camera...")
            from canvas import GestureCanvas
            shared_canvas = GestureCanvas(num_hands=1, allow_color_change=True, allow_clear=True)
            for _ in range(5):  # Warm up the camera before we start using it
                shared_canvas.update()
        return shared_canvas

    # Import the game modes here to avoid circular imports and to ensure they have access to the shared canvas when they are run
    from modes import run_free_draw, run_solo_challenge, run_duel

    try:
        while True:
            # Run the menu and get the user's choice
            choice, last_menu_frame = run_menu(window_name=WINDOW_NAME, fade_in_first=fade_in_menu_next)
            fade_in_menu_next = False

            # Handle the user's choice and launch the corresponding game mode
            if choice in ("free", "solo", "duel"):
                fade_out(WINDOW_NAME, last_menu_frame)

                if choice == "free":
                    canvas = get_canvas()
                    last_mode_frame = run_free_draw(canvas, window_name=WINDOW_NAME)
                elif choice == "solo":
                    canvas = get_canvas()
                    last_mode_frame = run_solo_challenge(canvas, window_name=WINDOW_NAME)
                elif choice == "duel":
                    canvas = get_canvas()
                    last_mode_frame = run_duel(canvas, window_name=WINDOW_NAME)

                if last_mode_frame is not None:
                    fade_out(WINDOW_NAME, last_mode_frame)

                fade_in_menu_next = True

            elif choice == "quit" or choice is None:
                fade_out(WINDOW_NAME, last_menu_frame)
                print("Bye!")
                break
    finally:
        # Clean up resources and close the OpenCV window when exiting the game
        if shared_canvas is not None:
            shared_canvas.close()
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()