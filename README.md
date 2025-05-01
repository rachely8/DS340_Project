# rock-paper-scissors

![visitors](https://visitor-badge.glitch.me/badge?page_id=SouravJohar.visitor-badge)

https://youtu.be/0uSA3xyXlwM 

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3.11.4 (install via Homebrew on macOS: `brew install python@3.11`)
- TensorFlow 2.x (macOS: `tensorflow-macos` + `tensorflow-metal` for GPU support)
- Pillow
- OpenCV

## Set up instructions

1. Create and activate the virtual environment (from the project root):
   ```sh
   # macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Gather images:
   ```sh
   python gather_images.py <gesture> <count>
   ```
   > **Note:** You can capture custom datasets by running:
   > ```sh
   > python gather_images.py <gesture> <count>
   > ```
   > where `<gesture>` is one of `rock`, `paper`, `scissors`, or `none`, and `<count>` is how many images to capture.

4. Train the model:
   ```sh
   python train.py
   ```

5. Test the model:
   ```sh
   python test.py <path_to_image>
   ```

6. Play the game:
   ```sh
   python play.py
   ```
   **Usage:** Press **a** to start the game and **q** to quit.

## What We Did
- We forked the original Rock-Paper-Scissors code and completely overhauled the training pipeline.
- Replaced the outdated SqueezeNet setup with a modern **EfficientNetB0** backbone pre-trained on ImageNet.
- Implemented directory-based data loading with `flow_from_directory(validation_split=0.2)` for an automatic 80/20 train/validation split.
- Added extensive data augmentation (rotation, shifts, zoom, flips, brightness adjustments) to improve robustness.
- Trained in two stages: freezing the EfficientNet base, then fine-tuning the top layers for our specific hand gestures.
- Provided updated `train.py` and `test.py` scripts compatible with TensorFlow 2.x on macOS (Apple M1/M2).
- Simplified setup to use first-party TensorFlow APIs—no more custom SqueezeNet monkey-patches.
- Updated `play.py` to use the EfficientNetB0 model for real-time webcam inference with proper preprocessing.
