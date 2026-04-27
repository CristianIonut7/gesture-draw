import json
import random


WORDS_PATH = "assets/words.json"


def load_words():
    with open(WORDS_PATH, "r") as f:
        return json.load(f)


def calculate_score(time_seconds, penalty=0):
    base = max(20, 100 - time_seconds * 2.7)
    final = base - penalty
    return max(0, int(round(final)))


def get_penalty_for_attempt(attempt_number):
    return 5 * (attempt_number + 1)


def check_word_match(predictions_top5, target_word):
    thresholds = [0.50, 0.25, 0.15]
    for i, (label, conf) in enumerate(predictions_top5[:3]):
        if label == target_word and conf >= thresholds[i]:
            return True
    return False


class SoloGame:
    def __init__(self, num_rounds=5):
        self.all_words = load_words()
        self.num_rounds = num_rounds
        self.current_round = 0
        self.scores = []
        self.used_words = set()
        self.current_word = None
        self.wrong_guesses = []
        self.total_penalty = 0
        self.attempt_count = 0

    def start_next_round(self):
        self.current_round += 1
        available = [w for w in self.all_words if w not in self.used_words]
        if not available:
            available = self.all_words
        self.current_word = random.choice(available)
        self.used_words.add(self.current_word)
        self.wrong_guesses = []
        self.total_penalty = 0
        self.attempt_count = 0
        return self.current_word

    def register_wrong_guess(self, guessed_label):
        penalty = get_penalty_for_attempt(self.attempt_count)
        self.attempt_count += 1
        self.total_penalty += penalty
        if guessed_label and guessed_label != self.current_word:
            self.wrong_guesses.append(guessed_label)
        return penalty

    def register_correct_guess(self, time_seconds):
        score = calculate_score(time_seconds, self.total_penalty)
        self.scores.append(score)
        return score

    def skip_round(self):
        self.scores.append(0)

    def is_finished(self):
        return self.current_round >= self.num_rounds

    def total_score(self):
        return sum(self.scores)


class DuelGame:
    def __init__(self, num_rounds=5):
        self.all_words = load_words()
        self.num_rounds = num_rounds
        self.current_round = 0
        self.used_words = set()
        self.current_word = None
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.round_winners = []

    def start_next_round(self):
        self.current_round += 1
        available = [w for w in self.all_words if w not in self.used_words]
        if not available:
            available = self.all_words
        self.current_word = random.choice(available)
        self.used_words.add(self.current_word)
        return self.current_word

    def register_winner(self, player):
        if player == "p1":
            self.wins_p1 += 1
        elif player == "p2":
            self.wins_p2 += 1
        self.round_winners.append(player)

    def register_skip(self):
        self.round_winners.append("none")

    def is_finished(self):
        return self.current_round >= self.num_rounds

    def overall_winner(self):
        if self.wins_p1 > self.wins_p2:
            return "p1"
        elif self.wins_p2 > self.wins_p1:
            return "p2"
        return "tie"