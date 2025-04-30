# import cv2
# import numpy as np
# from keras_squeezenet import SqueezeNet
# from keras.optimizers import Adam
# from keras.utils import np_utils
# from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
# from keras.models import Sequential
# import tensorflow as tf
# import os

# IMG_SAVE_PATH = 'image_data'

# CLASS_MAP = {
#     "rock": 0,
#     "paper": 1,
#     "scissors": 2,
#     "none": 3
# }

# NUM_CLASSES = len(CLASS_MAP)


# def mapper(val):
#     return CLASS_MAP[val]


# def get_model():
#     model = Sequential([
#         SqueezeNet(input_shape=(227, 227, 3), include_top=False),
#         Dropout(0.5),
#         Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
#         Activation('relu'),
#         GlobalAveragePooling2D(),
#         Activation('softmax')
#     ])
#     return model


# # load images from the directory
# dataset = []
# for directory in os.listdir(IMG_SAVE_PATH):
#     path = os.path.join(IMG_SAVE_PATH, directory)
#     if not os.path.isdir(path):
#         continue
#     for item in os.listdir(path):
#         # to make sure no hidden files get in our way
#         if item.startswith("."):
#             continue
#         img = cv2.imread(os.path.join(path, item))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (227, 227))
#         dataset.append([img, directory])

# '''
# dataset = [
#     [[...], 'rock'],
#     [[...], 'paper'],
#     ...
# ]
# '''
# data, labels = zip(*dataset)
# labels = list(map(mapper, labels))


# '''
# labels: rock,paper,paper,scissors,rock...
# one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
# '''

# # one hot encode the labels
# labels = np_utils.to_categorical(labels)

# # define the model
# model = get_model()
# model.compile(
#     optimizer=Adam(lr=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # start training
# model.fit(np.array(data), np.array(labels), epochs=10)

# # save the model for later use
# model.save("rock-paper-scissors-model.h5")

# import cv2
# import numpy as np
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications import MobileNetV2
# import os

# IMG_SAVE_PATH = 'image_data'

# CLASS_MAP = {
#     "rock": 0,
#     "paper": 1,
#     "scissors": 2,
#     "none": 3
# }

# NUM_CLASSES = len(CLASS_MAP)

# def mapper(val):
#     return CLASS_MAP[val]

# def get_model():
#     # Using MobileNetV2 as base model
#     base_model = MobileNetV2(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(224, 224, 3)
#     )
    
#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),
#         Dropout(0.5),
#         Dense(NUM_CLASSES, activation='softmax')
#     ])
    
#     # Freeze base model layers
#     base_model.trainable = False
#     return model

# # Load images (original comments preserved)
# dataset = []
# for directory in os.listdir(IMG_SAVE_PATH):
#     path = os.path.join(IMG_SAVE_PATH, directory)
#     if not os.path.isdir(path):
#         continue
#     for item in os.listdir(path):
#         if item.startswith("."):
#             continue
#         img = cv2.imread(os.path.join(path, item))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (224, 224))
#         dataset.append([img, directory])

# '''
# dataset = [
#     [[...], 'rock'],
#     [[...], 'paper'],
#     ...
# ]
# '''
# data, labels = zip(*dataset)
# labels = list(map(mapper, labels))

# # One hot encode
# labels = to_categorical(labels)
# data = np.array(data, dtype=np.float32) / 255.0

# # Create model
# model = get_model()
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train
# model.fit(data, labels, epochs=10, batch_size=32)

# # Save
# model.save("rock-paper-scissors-model.h5")
# print("Model saved successfully!")


# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dropout, Conv2D, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# import os

# # Keep original paths and class mapping
# IMG_SAVE_PATH = 'image_data'

# CLASS_MAP = {
#     "rock": 0,
#     "paper": 1,
#     "scissors": 2,
#     "none": 3
# }

# NUM_CLASSES = len(CLASS_MAP)

# def mapper(val):
#     return CLASS_MAP[val]

# def get_model():
#     """
#     Create a custom CNN model similar to SqueezeNet but compatible with current TensorFlow/Keras
#     """
#     model = Sequential()
    
#     # Create a simplified architecture inspired by SqueezeNet
#     # Input layer
#     model.add(Conv2D(64, (3, 3), padding='same', input_shape=(227, 227, 3)))
#     model.add(Activation('relu'))
    
#     # Fire module 1
#     model.add(Conv2D(16, (1, 1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (1, 1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
    
#     # Fire module 2
#     model.add(Conv2D(16, (1, 1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(128, (1, 1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(128, (3, 3), padding='same'))
#     model.add(Activation('relu'))
    
#     # Dropout layer
#     model.add(Dropout(0.5))
    
#     # Output layer
#     model.add(Conv2D(NUM_CLASSES, (1, 1), padding='valid'))
#     model.add(Activation('relu'))
#     model.add(GlobalAveragePooling2D())
#     model.add(Activation('softmax'))
    
#     return model

# # Load images from the directory
# print("Loading images...")
# dataset = []
# for directory in os.listdir(IMG_SAVE_PATH):
#     path = os.path.join(IMG_SAVE_PATH, directory)
#     if not os.path.isdir(path):
#         continue
#     for item in os.listdir(path):
#         # to make sure no hidden files get in our way
#         if item.startswith("."):
#             continue
#         img_path = os.path.join(path, item)
#         try:
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (227, 227))
#             dataset.append([img, directory])
#         except Exception as e:
#             print(f"Error loading {img_path}: {e}")

# # Check if dataset is empty
# if len(dataset) == 0:
#     print("No images found. Please run gather_images.py first to collect training data.")
#     exit()

# print(f"Total images loaded: {len(dataset)}")

# '''
# dataset = [
#     [[...], 'rock'],
#     [[...], 'paper'],
#     ...
# ]
# '''
# data, labels = zip(*dataset)
# labels = list(map(mapper, labels))

# '''
# labels: rock,paper,paper,scissors,rock...
# one hot encoded: [1,0,0], [0,1,0], [0,1,0], [0,0,1], [1,0,0]...
# '''

# # one hot encode the labels
# labels = to_categorical(labels)

# # Convert data to numpy array and normalize
# data = np.array(data, dtype="float") / 255.0

# # define the model
# print("Compiling model...")
# model = get_model()
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),  # Updated for TF 2.x compatibility
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Display model summary
# model.summary()

# # start training
# print("Starting training...")
# history = model.fit(
#     data, labels, 
#     batch_size=32,
#     epochs=10,
#     validation_split=0.2,  # Use 20% of data for validation
#     verbose=1
# )

# # save the model for later use
# print("Saving model...")
# model.save("rock-paper-scissors-model.h5")
# print("Model saved to rock-paper-scissors-model.h5")

# # Print final metrics
# print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
# if 'val_accuracy' in history.history:
#     print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# print("\nTraining complete!")

# import cv2
# import numpy as np
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# IMG_SAVE_PATH = 'image_data'

# CLASS_MAP = {
#     "rock": 0,
#     "paper": 1,
#     "scissors": 2,
#     "none": 3
# }

# NUM_CLASSES = len(CLASS_MAP)
# IMG_SIZE = 224  # EfficientNetB0 default input size
# BATCH_SIZE = 32

# def mapper(val):
#     return CLASS_MAP[val]

# def get_model():
#     # Using EfficientNetB0 as base model - more accurate than MobileNet/SqueezeNet
#     base_model = EfficientNetB0(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(IMG_SIZE, IMG_SIZE, 3)
#     )
    
#     model = Sequential([
#         base_model,
#         GlobalAveragePooling2D(),
#         Dropout(0.5),
#         Dense(NUM_CLASSES, activation='softmax')
#     ])
    
#     # Freeze base model layers initially
#     base_model.trainable = False
#     return model

# def load_data():
#     dataset = []
#     for directory in os.listdir(IMG_SAVE_PATH):
#         path = os.path.join(IMG_SAVE_PATH, directory)
#         if not os.path.isdir(path):
#             continue
#         for item in os.listdir(path):
#             if item.startswith("."):
#                 continue
#             img = cv2.imread(os.path.join(path, item))
#             if img is None:
#                 continue  # skip if image couldn't be loaded
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#             dataset.append([img, directory])
    
#     if not dataset:
#         raise ValueError("No images found in the dataset!")
    
#     data, labels = zip(*dataset)
#     labels = list(map(mapper, labels))
    
#     # One hot encode and normalize
#     labels = to_categorical(labels)
#     data = np.array(data, dtype=np.float32) / 255.0
    
#     return data, labels

# def train_model():
#     # Load data
#     data, labels = load_data()
    
#     # Create data augmentation generator
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest',
#         validation_split=0.2  # Using 20% for validation
#     )
    
#     # Create model
#     model = get_model()
    
#     # Compile with lower initial learning rate
#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Train with data augmentation
#     train_generator = datagen.flow(
#         data, labels,
#         batch_size=BATCH_SIZE,
#         subset='training')
    
#     validation_generator = datagen.flow(
#         data, labels,
#         batch_size=BATCH_SIZE,
#         subset='validation')
    
#     # First stage training - frozen base
#     print("First stage training - frozen base model")
#     history = model.fit(
#         train_generator,
#         steps_per_epoch=len(train_generator),
#         validation_data=validation_generator,
#         validation_steps=len(validation_generator),
#         epochs=10
#     )
    
#     # Unfreeze some layers for fine-tuning
#     model.layers[0].trainable = True
#     for layer in model.layers[0].layers[:len(model.layers[0].layers)//2]:
#         layer.trainable = False
    
#     # Recompile with lower learning rate for fine-tuning
#     model.compile(
#         optimizer=Adam(learning_rate=0.00001),  # lower LR for fine-tuning
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Second stage training - fine-tuning
#     print("\nSecond stage training - fine-tuning")
#     history = model.fit(
#         train_generator,
#         steps_per_epoch=len(train_generator),
#         validation_data=validation_generator,
#         validation_steps=len(validation_generator),
#         epochs=10
#     )
    
#     # Save the model
#     model.save("rock-paper-scissors-model.h5")
#     print("Model saved successfully!")
    
#     return model

# if __name__ == "__main__":
#     model = train_model()

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers, Model

# ---- Config ----
IMG_DIR      = 'image_data'
IMG_SIZE     = (224, 224)       # EfficientNetB0 default
BATCH_SIZE   = 32
EPOCHS       = 20
NUM_CLASSES  = 4                # rock, paper, scissors, none

# ---- Build model ----
def build_model():
    # 1) Load EfficientNetB0 without its top, with ImageNet weights
    base = EfficientNetB0(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # freeze backbone

    # 2) Add classification head
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=base.input, outputs=outputs, name='rps_efficientnetb0')

# ---- Data pipeline ----
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # applies EfficientNet’s preprocessing
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    IMG_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)
val_gen = datagen.flow_from_directory(
    IMG_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# ---- Compile & train ----
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ---- Fine-tune last layers ----
# Unfreeze top 20% of layers in the model
total_layers = len(model.layers)
num_unfreeze = int(total_layers * 0.2)
for layer in model.layers[:-num_unfreeze]:
    layer.trainable = False
for layer in model.layers[-num_unfreeze:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# ---- Save final model ----
model.save('rps_efficientnetb0_final.keras')
print("Training complete — model saved as rps_efficientnetb0_final.keras")