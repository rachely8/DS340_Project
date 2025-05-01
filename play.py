import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import numpy as np
from random import choice

# Win/loss/tie tracking
user_wins = 0
computer_wins = 0
ties = 0

# Markov model for prediction of user moves
history = []
k = 3  # order of the model
transition_counts = {}
# mapping human move to counter move
counter = {'rock':'paper','paper':'scissors','scissors':'rock'}

DECAY = 0.99  # exponential decay factor for recency weighting
transition_counts_1 = {}  # order-1 transition counts
transition_counts_2 = {}  # order-2 transition counts
transition_counts_3 = {}  # order-3 transition counts

REV_CLASS_MAP = {
    0: "none",
    1: "paper",
    2: "rock",
    3: "scissors"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = load_model("rps_efficientnetb0_final.keras")

cap = cv2.VideoCapture(0)


# Game state variables
game_state = "waiting"     # one of "waiting", "countdown", "result"
countdown_start = None
countdown_secs = 3         # seconds for the countdown
user_move_name = "none"
computer_move_name = "none"
winner = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    k = cv2.waitKey(10)
    # Handle key presses
    if k == ord(' '):  # SPACE starts a new round or resets after result
        if game_state == "waiting":
            countdown_start = time.time()
            game_state = "countdown"
        elif game_state == "result":
            game_state = "waiting"
            user_move_name = "none"
            computer_move_name = "none"
            winner = ""
    if k == ord('q'):
        break

    current_time = time.time()
    if game_state == "countdown":
        elapsed = current_time - countdown_start
        if elapsed < countdown_secs:
            # display countdown number
            num = countdown_secs - int(elapsed)
            cv2.putText(frame, str(num),
                        (600, 350), cv2.FONT_HERSHEY_SIMPLEX,
                        5, (0, 0, 255), 10, cv2.LINE_AA)
        else:
            # capture ROI and make one prediction
            roi = frame[100:500, 100:500]
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32')
            img = preprocess_input(img)
            pred = model.predict(np.expand_dims(img, axis=0))
            move_code = np.argmax(pred[0])
            user_move_name = mapper(move_code)
            # computer and winner
            if user_move_name != "none":
                # apply exponential decay to prioritize recent patterns
                for counts in transition_counts_2.values():
                    for move in counts:
                        counts[move] *= DECAY
                for counts in transition_counts_1.values():
                    for move in counts:
                        counts[move] *= DECAY
                for counts in transition_counts_3.values():
                    for move in counts:
                        counts[move] *= DECAY

                # update order-2 counts
                if len(history) >= 2:
                    key2 = tuple(history[-2:])
                    transition_counts_2.setdefault(key2, {'rock':0,'paper':0,'scissors':0})
                    transition_counts_2[key2][user_move_name] += 1

                # update order-1 counts
                if len(history) >= 1:
                    key1 = (history[-1],)
                    transition_counts_1.setdefault(key1, {'rock':0,'paper':0,'scissors':0})
                    transition_counts_1[key1][user_move_name] += 1

                # update order-3 counts
                if len(history) >= 3:
                    key3 = tuple(history[-3:])
                    transition_counts_3.setdefault(key3, {'rock':0, 'paper':0, 'scissors':0})
                    transition_counts_3[key3][user_move_name] += 1

                history.append(user_move_name)
                # ensemble Markov prediction: try order-3, then order-2, then order-1, then random
                predicted = None
                # first try order-3 Markov
                if len(history) >= 3 and predicted is None:
                    key3 = tuple(history[-3:])
                    counts3 = transition_counts_3.get(key3, {})
                    if counts3 and sum(counts3.values()) > 0:
                        predicted = max(counts3, key=counts3.get)
                if len(history) >= 2:
                    key2 = tuple(history[-2:])
                    counts2 = transition_counts_2.get(key2, {})
                    if counts2 and sum(counts2.values()) > 0:
                        predicted = max(counts2, key=counts2.get)
                if predicted is None and len(history) >= 1:
                    key1 = (history[-1],)
                    counts1 = transition_counts_1.get(key1, {})
                    if counts1 and sum(counts1.values()) > 0:
                        predicted = max(counts1, key=counts1.get)
                if predicted is None:
                    predicted = choice(['rock','paper','scissors'])
                # choose counter move
                computer_move_name = counter[predicted]
                # determine winner
                winner = calculate_winner(user_move_name, computer_move_name)
                # update score tracker
                if winner == "User":
                    user_wins += 1
                elif winner == "Computer":
                    computer_wins += 1
                else:
                    ties += 1
            else:
                computer_move_name = "none"
                winner = "No play detected"
            game_state = "result"

    # draw play areas
    cv2.rectangle(frame, (100,100),(500,500),(255,255,255),2)
    cv2.rectangle(frame, (800,100),(1200,500),(255,255,255),2)
    # overlay icon only after result
    if game_state == "result" and computer_move_name != "none":
        icon = cv2.imread(f"images/{computer_move_name}.png")
        icon = cv2.resize(icon, (400,400))
        frame[100:500, 800:1200] = icon

    # display the information always, but label differently by state
    font = cv2.FONT_HERSHEY_SIMPLEX
    if game_state == "result":
        cv2.putText(frame, "Your Move: " + user_move_name, (50,50), font, 1.2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Comp Move: " + computer_move_name, (750,50), font, 1.2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Winner: " + winner, (400,600), font, 2, (0,0,255), 4, cv2.LINE_AA)
    else:
        prompt = "[SPACE] to Play" if game_state=="waiting" else "Get Ready..."
        cv2.putText(frame, prompt, (400,600), font, 1.5, (0,255,255), 3, cv2.LINE_AA)

    # display win tracker
    cv2.putText(frame,
                f"Score - You: {user_wins}  Comp: {computer_wins}  Ties: {ties}",
                (50, 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA)
    cv2.imshow("Rock Paper Scissors", frame)

cap.release()
cv2.destroyAllWindows()
