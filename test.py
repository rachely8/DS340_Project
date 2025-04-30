from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys
from tensorflow.keras.applications.efficientnet import preprocess_input

filepath = sys.argv[1]

# Inferred from flow_from_directory: folders sorted as ['none','paper','rock','scissors']
REV_CLASS_MAP = {
    0: "none",
    1: "paper",
    2: "rock",
    3: "scissors"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rps_efficientnetb0_final.keras")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.array(img, dtype=np.float32)
img = preprocess_input(img)

# predict the move made
pred = model.predict(np.expand_dims(img, axis=0))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
