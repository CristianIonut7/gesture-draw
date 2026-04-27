""" This module defines the core game logic for both Solo and Duel modes. It handles word selection, scoring, penalties, and game state management."""

# json is used for loading the list of words and saving/loading game state if needed
# random is used for selecting random words for each round
import json
import random


WORDS_PATH = "assets/words.json"

# Scoring and penalty parameters
BASE_SCORE = 100
TIME_FACTOR = 2.7
PENALTY_PER_ATTEMPT = 5

# Load the list of words
def load_words():
    with open(WORDS_PATH, "r") as f:
        return json.load(f) #convert to list if needed

# Score is based on how quickly the player guesses the word, with a penalty for each wrong attempt
def calculate_score(time_seconds, penalty=0):
    base = max(20, BASE_SCORE - time_seconds * TIME_FACTOR)
    final = base - penalty
    return max(0, int(round(final)))

# Penalty increases with each wrong attempt
def get_penalty_for_attempt(attempt_number):
    return PENALTY_PER_ATTEMPT * (attempt_number + 1)

# Checks if the target word is among the top 5 predictions
def check_word_match(predictions_top5, target_word):
    thresholds = [0.50, 0.25, 0.10] # confidence thresholds for top 3 predictions
    for i, (label, conf) in enumerate(predictions_top5[:3]):
        if label == target_word and conf >= thresholds[i]:
            return True
    return False

class SoloGame:
    # Constructor initializes the game and resets the game state
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

    # start_next_round selects a new word for the player to draw
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

    # register_wrong_guess updates the game state when the player makes an incorrect guess
    def register_wrong_guess(self, guessed_label):
        penalty = get_penalty_for_attempt(self.attempt_count)
        self.attempt_count += 1
        self.total_penalty += penalty
        if guessed_label and guessed_label != self.current_word:
            self.wrong_guesses.append(guessed_label)
        return penalty

    # register_correct_guess calculates the score for a correct guess
    def register_correct_guess(self, time_seconds):
        score = calculate_score(time_seconds, self.total_penalty)
        self.scores.append(score)
        return score

    # round skipping allows the player to forfeit the current word and move on to the next one, but with a score of 0 for that round
    def skip_round(self):
        self.scores.append(0)

    # is_finished checks if the game has reached the maximum number of rounds
    def is_finished(self):
        return self.current_round >= self.num_rounds

    # total_score sums up the scores from all rounds
    def total_score(self):
        return sum(self.scores)

class DuelGame:
    # Constructor initializes the game and resets the game state for both players
    def __init__(self, num_rounds=5):
        self.all_words = load_words()
        self.num_rounds = num_rounds
        self.current_round = 0
        self.used_words = set()
        self.current_word = None
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.round_winners = []

    # start_next_round selects a new word for both players to draw
    def start_next_round(self):
        self.current_round += 1
        available = [w for w in self.all_words if w not in self.used_words]
        if not available:
            available = self.all_words
        self.current_word = random.choice(available)
        self.used_words.add(self.current_word)
        return self.current_word

    # register_winner updates the game state when a player wins a round
    def register_winner(self, player):
        if player == "p1":
            self.wins_p1 += 1
        elif player == "p2":
            self.wins_p2 += 1
        self.round_winners.append(player)

    # register_skip allows both players to forfeit the current round
    def register_skip(self):
        self.round_winners.append("none")

    # is_finished checks if the game has reached the maximum number of rounds
    def is_finished(self):
        return self.current_round >= self.num_rounds

    # overall_winner determines the winner
    def overall_winner(self):
        if self.wins_p1 > self.wins_p2:
            return "p1"
        elif self.wins_p2 > self.wins_p1:
            return "p2"
        return "tie"