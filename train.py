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