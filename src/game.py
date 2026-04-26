"""
game.py
Game logic: word selection, scoring, round management.
Independent of UI / camera - just pure logic.
"""
import json
import random


WORDS_PATH = "assets/words.json"


def load_words():
    """Load list of words available for the game."""
    with open(WORDS_PATH, "r") as f:
        return json.load(f)


def calculate_score(time_seconds, penalty=0):
    """
    Score formula: starts at 100, drops linearly to a min of 20 by 30s.
    After that, stays at 20 (still rewards finishing slow).
    Penalty is subtracted at the end.
    """
    base = max(20, 100 - time_seconds * 2.7)
    final = base - penalty
    return max(0, int(round(final)))


def get_penalty_for_attempt(attempt_number):
    """
    Increasing penalty: 1st wrong = 5, 2nd = 10, 3rd = 15, ...
    attempt_number is the index of this wrong guess (0-based).
    Returns the cost of THIS specific wrong guess.
    """
    return 5 * (attempt_number + 1)


class SoloGame:
    """
    Manages a Solo Challenge session of N rounds.
    """

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
        """Pick a new word and reset round state. Returns the new word."""
        self.current_round += 1
        available = [w for w in self.all_words if w not in self.used_words]
        if not available:
            available = self.all_words  # in case we run out
        self.current_word = random.choice(available)
        self.used_words.add(self.current_word)
        self.wrong_guesses = []
        self.total_penalty = 0
        self.attempt_count = 0
        return self.current_word

    def register_wrong_guess(self, guessed_label):
        """AI guessed wrong. Returns penalty for this attempt."""
        penalty = get_penalty_for_attempt(self.attempt_count)
        self.attempt_count += 1
        self.total_penalty += penalty
        if guessed_label and guessed_label != self.current_word:
            self.wrong_guesses.append(guessed_label)
        return penalty

    def register_correct_guess(self, time_seconds):
        """AI guessed right. Compute final score for the round."""
        score = calculate_score(time_seconds, self.total_penalty)
        self.scores.append(score)
        return score

    def skip_round(self):
        """User skipped - 0 points for this round."""
        self.scores.append(0)

    def is_finished(self):
        return self.current_round >= self.num_rounds

    def total_score(self):
        return sum(self.scores)