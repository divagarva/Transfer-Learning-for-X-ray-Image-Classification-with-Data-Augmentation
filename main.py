import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import MobileNetV2, DenseNet169
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import preprocess_input

# Set constants
CLASSES = 2
WIDTH, HEIGHT = 224, 224
BATCH_SIZE = 16
EPOCHS = 5

# Define paths
PATH = os.path.join(os.getcwd(), "chest_xray")
TRAIN_DIR = os.path.join(PATH, 'train')
VAL_DIR = os.path.join(PATH, 'val')
TEST_DIR = os.path.join(PATH, 'test')

# Load base models
input_shape = (WIDTH, HEIGHT, 3)
input_layer = Input(shape=input_shape)

mobilenet_base = MobileNetV2(weights='imagenet', input_shape=input_shape, include_top=False)
densenet_base = DenseNet169(weights='imagenet', input_shape=input_shape, include_top=False)

# Freeze base model layers
for layer in mobilenet_base.layers:
    layer.trainable = False

for layer in densenet_base.layers:
    layer.trainable = False

# Create custom stacked model
model_mobilenet = mobilenet_base(input_layer)
model_mobilenet2 = GlobalAveragePooling2D()(model_mobilenet)
output_mobilenet = Flatten()(model_mobilenet2)

model_densenet = densenet_base(input_layer)
model_densenet2 = GlobalAveragePooling2D()(model_densenet)
output_densenet = Flatten()(model_densenet2)

merged = Concatenate()([output_mobilenet, output_densenet])

x = BatchNormalization()(merged)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(CLASSES, activation='softmax')(x)

stacked_model = Model(inputs=input_layer, outputs=x)

# Compile model
stacked_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
stacked_model.summary()

# Data preparation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Adjust steps based on dataset size to prevent `OUT_OF_RANGE` errors
STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size
VALIDATION_STEPS = max(1, validation_generator.samples // validation_generator.batch_size)

# Train model
stacked_model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    verbose=1
)

# Predict on a sample image
img_path = os.path.join(TEST_DIR, 'PNEUMONIA', 'person91_bacteria_445.jpeg')
img = load_img(img_path, target_size=(WIDTH, HEIGHT))  # Resize to 224x224 to match the model's input size
x = img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

# Make a prediction
pred = stacked_model.predict(x)[0]
print("Prediction:", pred)
print("Class Indices:", train_generator.class_indices)
