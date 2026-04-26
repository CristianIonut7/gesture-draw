import time
import cv2
import numpy as np

from game import SoloGame, DuelGame, check_word_match
from classifier import DrawingClassifier
from ui import make_paper_background, COLOR_INK, COLOR_INK_LIGHT, DISPLAY_W, DISPLAY_H


def _show_simple_loading(window_name, message):
    bg = make_paper_background(DISPLAY_W, DISPLAY_H)
    title = "GestureWar"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.2, 2)[0]
    cv2.putText(bg, title, ((DISPLAY_W - title_size[0]) // 2, DISPLAY_H // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_INK, 2, cv2.LINE_AA)
    msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
    cv2.putText(bg, message, ((DISPLAY_W - msg_size[0]) // 2, DISPLAY_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INK_LIGHT, 1, cv2.LINE_AA)
    cv2.imshow(window_name, bg)
    cv2.waitKey(1)
    return bg


def run_free_draw(canvas, window_name="GestureWar"):
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
    while True:
        state = canvas.update()
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

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
    preds = classifier.predict(raw_canvas, top_k=10)
    if preds is None:
        return None
    for label, conf in preds:
        if label not in banned_labels:
            return (label, conf)
    return None


def _show_round_result(canvas, window_name, success, word, score_change, total_score, duration_sec=2.5):
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


def _show_solo_final_score(canvas, window_name, scores, total):
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

            cv2.rectangle(display, (0, 0), (w, 70), (20, 20, 20), -1)
            cv2.putText(display, f"Draw: {word.upper()}", (20, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(display, f"Round {game.current_round}/{game.num_rounds}",
                        (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, f"Score: {game.total_score()}",
                        (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

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

    return _show_solo_final_score(canvas, window_name, game.scores, game.total_score())


def _duel_show_tutorial(duel_canvas, window_name):
    while True:
        state = duel_canvas.update(drawing_enabled=False, allow_clear=False)
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        title = "How to play - 1v1 Battle"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        lines = [
            "1. Two players, one camera. Player 1 = LEFT, Player 2 = RIGHT",
            "2. Each player draws on their own half of the screen",
            "3. The same word appears for both - first one recognized wins!",
            "4. AI checks every ~1.2 seconds. Be quick AND clear",
            "5. Make a fist on your side to clear your canvas",
            "6. Best of 5 rounds. Press S to skip a round if too hard",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (w // 2 - 380, 170 + i * 50),
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


def _duel_show_round_intro(duel_canvas, window_name, round_num, total_rounds, word,
                            wins_p1, wins_p2):
    last_display = None
    start = time.time()
    while time.time() - start < 1.5:
        state = duel_canvas.update(drawing_enabled=False, allow_clear=False)
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        round_text = f"Round {round_num} / {total_rounds}"
        rt_size = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 1)[0]
        cv2.putText(display, round_text, ((w - rt_size[0]) // 2, h // 2 - 100),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 1, cv2.LINE_AA)

        word_text = f"Draw  a  {word.upper()}"
        wt_size = cv2.getTextSize(word_text, cv2.FONT_HERSHEY_DUPLEX, 2.2, 3)[0]
        cv2.putText(display, word_text, ((w - wt_size[0]) // 2, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 3, cv2.LINE_AA)

        score_text = f"P1: {wins_p1}    -    P2: {wins_p2}"
        st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
        cv2.putText(display, score_text, ((w - st_size[0]) // 2, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        cv2.waitKey(1)

    countdown_start = time.time()
    countdown_total = 3.5
    while time.time() - countdown_start < countdown_total:
        state = duel_canvas.update(drawing_enabled=False, allow_clear=False)
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        cv2.putText(display, f"Draw  a  {word.upper()}", (w // 2 - 200, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        elapsed = time.time() - countdown_start
        if elapsed < 1.0:
            cd_text = "3"
        elif elapsed < 2.0:
            cd_text = "2"
        elif elapsed < 3.0:
            cd_text = "1"
        else:
            cd_text = "START!"

        cd_color = (100, 255, 200) if cd_text == "START!" else (255, 255, 255)
        cd_scale = 8.0 if cd_text != "START!" else 4.0
        cd_size = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_DUPLEX, cd_scale, 6)[0]
        cv2.putText(display, cd_text, ((w - cd_size[0]) // 2, h // 2 + cd_size[1] // 2),
                    cv2.FONT_HERSHEY_DUPLEX, cd_scale, cd_color, 6, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        cv2.waitKey(1)

    return last_display


def _duel_show_round_result(duel_canvas, window_name, winner, word, wins_p1, wins_p2,
                             duration_sec=2.5):
    start = time.time()
    last_display = None

    if winner == "p1":
        title = "PLAYER 1 WINS!"
        color = (255, 200, 100)
    elif winner == "p2":
        title = "PLAYER 2 WINS!"
        color = (100, 180, 255)
    else:
        title = "ROUND SKIPPED"
        color = (180, 180, 180)

    while time.time() - start < duration_sec:
        state = duel_canvas.update(drawing_enabled=False, allow_clear=False)
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, h // 2 - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 3, cv2.LINE_AA)

        sub = f"Word was: {word.upper()}"
        sub_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
        cv2.putText(display, sub, ((w - sub_size[0]) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 1, cv2.LINE_AA)

        score_text = f"P1: {wins_p1}    -    P2: {wins_p2}"
        st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        cv2.putText(display, score_text, ((w - st_size[0]) // 2, h // 2 + 90),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        cv2.waitKey(1)

    return last_display


def _duel_show_final(duel_canvas, window_name, game):
    last_display = None
    overall = game.overall_winner()
    if overall == "p1":
        title = "PLAYER 1 WINS THE GAME!"
        color = (255, 200, 100)
    elif overall == "p2":
        title = "PLAYER 2 WINS THE GAME!"
        color = (100, 180, 255)
    else:
        title = "IT'S A TIE!"
        color = (220, 220, 220)

    while True:
        state = duel_canvas.update(drawing_enabled=False, allow_clear=False)
        if state is None:
            return None
        display = state["display"]
        h, w, _ = display.shape

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)

        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
        cv2.putText(display, title, ((w - title_size[0]) // 2, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 3, cv2.LINE_AA)

        score_text = f"Final Score:  P1 {game.wins_p1}  -  P2 {game.wins_p2}"
        st_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        cv2.putText(display, score_text, ((w - st_size[0]) // 2, 210),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        for i, w_id in enumerate(game.round_winners):
            if w_id == "p1":
                line = f"Round {i+1}:  Player 1"
                lc = (255, 200, 100)
            elif w_id == "p2":
                line = f"Round {i+1}:  Player 2"
                lc = (100, 180, 255)
            else:
                line = f"Round {i+1}:  skipped"
                lc = (160, 160, 160)
            cv2.putText(display, line, (w // 2 - 100, 280 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, lc, 1, cv2.LINE_AA)

        cv2.putText(display, "Press SPACE or ESC to return to menu",
                    (w // 2 - 220, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        last_display = display
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or key == ord('q') or key == 27:
            return last_display


def run_duel(shared_canvas, window_name="GestureWar"):
    
    _show_simple_loading(window_name, "preparing 1v1 mode...")
    shared_canvas.cap.release()

    from duel_canvas import DuelCanvas
    duel_canvas = DuelCanvas()
    for _ in range(5):
        duel_canvas.update(drawing_enabled=False, allow_clear=False)

    classifier = DrawingClassifier()
    last_display = None

    try:
        tutorial_result = _duel_show_tutorial(duel_canvas, window_name)
        if tutorial_result is None:
            return None
        last_display, action = tutorial_result
        if action == "back":
            return last_display

        game = DuelGame(num_rounds=5)

        while not game.is_finished():
            word = game.start_next_round()
            duel_canvas.reset_canvases()

            last_display = _duel_show_round_intro(duel_canvas, window_name,
                                                   game.current_round, game.num_rounds,
                                                   word, game.wins_p1, game.wins_p2)

            round_winner = None
            last_ai_check = time.time()
            ai_check_interval = 1.2
            p1_match_streak = 0
            p2_match_streak = 0
            REQUIRED_STREAK = 2

            while round_winner is None:
                state = duel_canvas.update(drawing_enabled=True, allow_clear=True)
                if state is None:
                    return last_display

                display = state["display"]
                h, w, _ = display.shape

                cv2.rectangle(display, (0, 0), (w, 60), (20, 20, 20), -1)
                cv2.putText(display, f"Draw: {word.upper()}", (20, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display, f"Round {game.current_round}/{game.num_rounds}",
                            (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display, f"P1 {game.wins_p1}  -  P2 {game.wins_p2}",
                            (w - 200, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.putText(display, "PLAYER 1", (30, h - 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 100), 2, cv2.LINE_AA)
                cv2.putText(display, "PLAYER 2", (w - 180, h - 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 180, 255), 2, cv2.LINE_AA)

                for i in range(REQUIRED_STREAK):
                    color_on = (200, 255, 200)
                    color_off = (60, 60, 60)
                    cv2.circle(display, (110 + i * 22, h - 50),
                               6, color_on if i < p1_match_streak else color_off, -1)
                    cv2.circle(display, (w - 100 + i * 22, h - 50),
                               6, color_on if i < p2_match_streak else color_off, -1)

                cv2.putText(display, "Fist = clear (your side)  |  S = skip  |  ESC = quit",
                            (w // 2 - 200, h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

                cv2.imshow(window_name, display)
                last_display = display

                if time.time() - last_ai_check >= ai_check_interval:
                    last_ai_check = time.time()

                    preds_p1 = classifier.predict(state["raw_canvas_p1"], top_k=5)
                    preds_p2 = classifier.predict(state["raw_canvas_p2"], top_k=5)

                    if preds_p1 and check_word_match(preds_p1, word):
                        p1_match_streak += 1
                    else:
                        p1_match_streak = 0

                    if preds_p2 and check_word_match(preds_p2, word):
                        p2_match_streak += 1
                    else:
                        p2_match_streak = 0

                    if p1_match_streak >= REQUIRED_STREAK and p2_match_streak >= REQUIRED_STREAK:
                        c1 = next((c for l, c in preds_p1 if l == word), 0)
                        c2 = next((c for l, c in preds_p2 if l == word), 0)
                        round_winner = "p1" if c1 >= c2 else "p2"
                    elif p1_match_streak >= REQUIRED_STREAK:
                        round_winner = "p1"
                    elif p2_match_streak >= REQUIRED_STREAK:
                        round_winner = "p2"

                    if round_winner:
                        game.register_winner(round_winner)
                        last_display = _duel_show_round_result(
                            duel_canvas, window_name, round_winner, word,
                            game.wins_p1, game.wins_p2)
                        break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    game.register_skip()
                    last_display = _duel_show_round_result(
                        duel_canvas, window_name, "none", word,
                        game.wins_p1, game.wins_p2)
                    round_winner = "none"
                    break
                elif key == ord('q') or key == 27:
                    return last_display

        last_display = _duel_show_final(duel_canvas, window_name, game)
        return last_display

    finally:
        _show_simple_loading(window_name, "returning to menu...")
        duel_canvas.close()
        shared_canvas.cap = cv2.VideoCapture(0)
        for _ in range(3):
            shared_canvas.update()