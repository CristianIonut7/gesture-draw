"""
modes.py
Game modes - functions called by main.py based on menu selection.
"""
import time
import cv2
import numpy as np

from game import SoloGame
from classifier import DrawingClassifier


def run_free_draw(canvas, window_name="GestureWar"):
    """Free Draw mode - draw whatever you want."""
    canvas.set_modes(allow_color_change=True, allow_clear=True)
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


def _show_tutorial(canvas, window_name):
    """
    Quick tutorial screen - returns last frame for fade-out.
    User presses any key to continue.
    """
    while True:
        state = canvas.update()
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        # Semi-transparent dark overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Tutorial text
        title = "How to play - Solo Challenge"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        lines = [
            "1. You will see a word at the top - draw it!",
            "2. Index finger = draw  |  Pinch = pen up  |  Fist = clear",
            "3. Open your hand to ask the AI to guess",
            "4. Faster = more points. Wrong AI guess = penalty + cooldown",
            "5. Wrong guesses are banned in next AI call this round",
            "6. 5 rounds. Good luck!",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (w // 2 - 380, 180 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(display, "Press SPACE to start  |  ESC to go back",
                    (w // 2 - 220, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return display, "start"
        elif key == ord('q') or key == 27:
            return display, "back"


def _show_round_intro(canvas, window_name, round_num, total_rounds, word, total_score, duration_sec=2.0):
    """Show 'Round N - Draw a STAR' for a few seconds."""
    start = time.time()
    last_display = None
    while time.time() - start < duration_sec:
        state = canvas.update()
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        round_text = f"Round {round_num} / {total_rounds}"
        rt_size = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 1)[0]
        cv2.putText(display, round_text, ((w - rt_size[0]) // 2, h // 2 - 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 1, cv2.LINE_AA)

        word_text = f"Draw  a  {word.upper()}"
        wt_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)[0]
        cv2.putText(display, word_text, ((w - wt_size[0]) // 2, h // 2 + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 3, cv2.LINE_AA)

        score_text = f"Total: {total_score}"
        st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.putText(display, score_text, ((w - st_size[0]) // 2, h // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            return last_display

    return last_display


def _classify_with_bans(classifier, raw_canvas, banned_labels):
    """Get top prediction excluding banned labels."""
    preds = classifier.predict(raw_canvas, top_k=10)
    if preds is None:
        return None
    for label, conf in preds:
        if label not in banned_labels:
            return (label, conf)
    return None


def _show_round_result(canvas, window_name, success, word, score_change, total_score, duration_sec=2.5):
    """Display round result: success or skip."""
    start = time.time()
    last_display = None
    color = (100, 255, 100) if success else (100, 100, 255)
    title = "CORRECT!" if success else "SKIPPED"

    while time.time() - start < duration_sec:
        state = canvas.update()
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.5, 3)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, h // 2 - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, color, 3, cv2.LINE_AA)

        if success:
            sub = f"You drew {word.upper()}  -  +{score_change} pts"
        else:
            sub = f"The word was {word.upper()}"
        sub_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
        cv2.putText(display, sub, ((w - sub_size[0]) // 2, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 1, cv2.LINE_AA)

        total_text = f"Total: {total_score}"
        t_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        cv2.putText(display, total_text, ((w - t_size[0]) // 2, h // 2 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        cv2.waitKey(1)

    return last_display


def _show_final_score(canvas, window_name, scores, total):
    """End-of-game screen with breakdown. Wait for SPACE/ESC."""
    last_display = None
    while True:
        state = canvas.update()
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        title = "Game Over"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

        # Breakdown
        for i, score in enumerate(scores):
            line = f"Round {i+1}:  {score} pts"
            cv2.putText(display, line, (w // 2 - 100, 200 + i * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 1, cv2.LINE_AA)

        total_text = f"Total: {total}"
        tt_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 2)[0]
        cv2.putText(display, total_text, ((w - tt_size[0]) // 2, 200 + len(scores) * 45 + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (130, 255, 130), 2, cv2.LINE_AA)

        cv2.putText(display, "Press SPACE or ESC to return to menu",
                    (w // 2 - 220, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q') or key == 27:
            return last_display


def run_solo_challenge(canvas, window_name="GestureWar"):
    """Solo Challenge: 5 rounds, draw and let AI guess."""
    canvas.set_modes(allow_color_change=False, allow_clear=True)
    canvas.reset_canvas()

    classifier = DrawingClassifier()

    tutorial_result = _show_tutorial(canvas, window_name)
    if tutorial_result is None:
        return None
    last_display, action = tutorial_result
    if action == "back":
        return last_display

    game = SoloGame(num_rounds=5)
    last_display = None

    while not game.is_finished():
        word = game.start_next_round()
        canvas.reset_canvas()

        last_display = _show_round_intro(canvas, window_name,
                                          game.current_round, game.num_rounds,
                                          word, game.total_score())

        round_start = time.time()
        cooldown_until = 0.0
        last_wrong_guess_msg = None
        last_wrong_guess_time = 0.0

        round_done = False
        while not round_done:
            state = canvas.update()
            if state is None:
                return last_display

            display = state["display"]
            h, w, _ = display.shape
            elapsed = time.time() - round_start
            cooldown_remaining = max(0, cooldown_until - time.time())

            # Header bar
            cv2.rectangle(display, (0, 0), (w, 70), (20, 20, 20), -1)
            cv2.putText(display, f"Draw: {word.upper()}", (20, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(display, f"Round {game.current_round}/{game.num_rounds}",
                        (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, f"Score: {game.total_score()}",
                        (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # Timer
            timer_text = f"{elapsed:5.1f} s"
            cv2.putText(display, timer_text, (w // 2 - 60, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (180, 220, 255), 2, cv2.LINE_AA)

            if cooldown_remaining > 0:
                cv2.putText(display, f"AI cooldown: {cooldown_remaining:.1f}s",
                            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2, cv2.LINE_AA)
            elif state["open_hand_active"]:
                progress = state["open_hand_progress"]
                bar_w = 300
                bar_x = w // 2 - bar_w // 2
                cv2.rectangle(display, (bar_x, 90), (bar_x + bar_w, 110), (40, 40, 40), -1)
                cv2.rectangle(display, (bar_x, 90), (bar_x + int(bar_w * progress), 110),
                              (100, 255, 200), -1)
                cv2.putText(display, "Calling AI...", (w // 2 - 60, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 200), 1, cv2.LINE_AA)

            if last_wrong_guess_msg and time.time() - last_wrong_guess_time < 2.5:
                cv2.putText(display, last_wrong_guess_msg, (20, h - 100),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 255), 2, cv2.LINE_AA)

            if game.wrong_guesses:
                banned_text = "Banned: " + ", ".join(game.wrong_guesses)
                cv2.putText(display, banned_text, (20, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 200), 1, cv2.LINE_AA)

            cv2.putText(display, "Open hand = call AI  |  Fist = clear  |  S = skip  |  ESC = quit",
                        (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)
            last_display = display

            if state["open_hand_triggered"] and cooldown_remaining <= 0:
                pred = _classify_with_bans(classifier, state["raw_canvas"], game.wrong_guesses)
                if pred is None:
                    last_wrong_guess_msg = "Canvas is empty - draw something!"
                    last_wrong_guess_time = time.time()
                else:
                    label, conf = pred
                    if label == word:
                        score = game.register_correct_guess(elapsed)
                        last_display = _show_round_result(canvas, window_name, True, word, score,
                                                          game.total_score())
                        round_done = True
                    else:
                        penalty = game.register_wrong_guess(label)
                        last_wrong_guess_msg = f"AI guessed {label.upper()} (-{penalty} pts)"
                        last_wrong_guess_time = time.time()
                        cooldown_until = time.time() + 7.0

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                game.skip_round()
                last_display = _show_round_result(canvas, window_name, False, word, 0,
                                                  game.total_score())
                round_done = True
            elif key == ord('q') or key == 27:
                return last_display

    return _show_final_score(canvas, window_name, game.scores, game.total_score())



def run_duel(canvas, window_name="GestureWar"):
    """Placeholder - implemented after solo testing."""
    print("1v1 Battle - coming soon!")
    return None