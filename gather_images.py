desc = '''Script to gather data images with a particular label.

Usage: python gather_images.py <label_name> <num_samples>

The script will collect <num_samples> number of images and store them
in its own directory.

Only the portion of the image within the box displayed
will be captured and stored.

Press 'a' to start/pause the image collecting process.
Press 'q' to quit.

'''

import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser(
    description="Gather gesture images. Press 'a' to start/pause, 'q' to quit."
)
parser.add_argument("label_name", choices=["rock", "paper", "scissors", "none"],
                    help="Gesture label for images")
parser.add_argument("num_samples", type=int,
                    help="Number of images to capture (>=0). Use 0 for unlimited until 'q' is pressed")
parser.add_argument("--roi", type=int, nargs=4, metavar=('X','Y','W','H'),
                    default=[100,100,400,400],
                    help="Region of interest: X Y W H rectangle for capture")
args = parser.parse_args()

label_name = args.label_name
num_samples = args.num_samples
x, y, w, h = args.roi

IMG_SAVE_PATH = 'image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, label_name)
os.makedirs(IMG_CLASS_PATH, exist_ok=True)

# Determine starting count to append new images
existing = [f for f in os.listdir(IMG_CLASS_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if existing:
    nums = [int(f.rsplit('_', 1)[-1].split('.')[0]) for f in existing]
    count = max(nums) + 1
else:
    count = 0

start_count = count

cap = cv2.VideoCapture(0)

start = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    if start:
        roi = frame[y:y+h, x:x+w]
        save_path = os.path.join(IMG_CLASS_PATH, f"{label_name}_{count}.jpg")
        cv2.imwrite(save_path, roi)
        count += 1
        # Stop after capturing the requested number of new images
        if num_samples > 0 and (count - start_count) >= num_samples:
            break

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"[a] Start/Pause  [q] Quit   Collected: {count - start_count}/{num_samples or '∞'}",
                (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(10)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print(f"\n{count - start_count} new image(s) saved to {IMG_CLASS_PATH}")
cap.release()
cv2.destroyAllWindows()
