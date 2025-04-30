# rock-paper-scissors

![visitors](https://visitor-badge.glitch.me/badge?page_id=SouravJohar.visitor-badge)

https://youtu.be/0uSA3xyXlwM 

An AI to play the Rock Paper Scissors game

## Requirements
- Python 3.8+
- TensorFlow 2.x (macOS: `tensorflow-macos` + `tensorflow-metal` for GPU support)
- Pillow
- OpenCV

## Set up instructions
1. Install dependencies:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

2. Gather images:
```sh
python gather_images.py <gesture> <count>
```

3. Train the model:
```sh
python train.py
```

4. Test the model:
```sh
python test.py <path_to_image>
```

5. Play the game:
```sh
python play.py
```

```sh
git clone https://github.com/rachely8/DS340_Project.git
cd DS340_Project/rock-paper-scissors
```

## What We Did
- We forked the original Rock-Paper-Scissors code and completely overhauled the training pipeline.
- Replaced the outdated SqueezeNet setup with a modern **EfficientNetB0** backbone pre-trained on ImageNet.
- Implemented directory-based data loading with `flow_from_directory(validation_split=0.2)` for an automatic 80/20 train/validation split.
- Added extensive data augmentation (rotation, shifts, zoom, flips, brightness adjustments) to improve robustness.
- Trained in two stages: freezing the EfficientNet base, then fine-tuning the top layers for our specific hand gestures.
- Provided updated `train.py` and `test.py` scripts compatible with TensorFlow 2.x on macOS (Apple M1/M2).
- Simplified setup to use first-party TensorFlow APIs—no more custom SqueezeNet monkey-patches.
- Updated `play.py` to use the EfficientNetB0 model for real-time webcam inference with proper preprocessing.
