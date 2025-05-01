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
k = 2  # order of the model
transition_counts = {}
# mapping human move to counter move
counter = {'rock':'paper','paper':'scissors','scissors':'rock'}

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
                # update transition counts
                if len(history) >= k:
                    key = tuple(history[-k:])
                    transition_counts.setdefault(key, {'rock':0,'paper':0,'scissors':0})
                    transition_counts[key][user_move_name] = transition_counts[key].get(user_move_name, 0) + 1
                history.append(user_move_name)
                # predict next human move
                if len(history) >= k and tuple(history[-k:]) in transition_counts:
                    key = tuple(history[-k:])
                    predicted = max(transition_counts[key], key=transition_counts[key].get)
                else:
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
