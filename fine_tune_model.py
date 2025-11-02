import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load existing model
MODEL_PATH = "saved_models/ADEDIWURA_emotion_net.h5"
model = load_model(MODEL_PATH)

# Replace the last layer to match 7 classes
x = model.layers[-2].output  # take output from the layer before last
output = Dense(7, activation='softmax')(x)  # new output layer for 7 classes
model = Model(inputs=model.input, outputs=output)

# Dataset folder
DATASET_DIR = "emotion_dataset"

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Training and validation generators
train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(48, 48),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(48, 48),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Compile model with small learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10  # increase if you want
)

# Save as a new model
model.save("saved_models/ADEDIWURA_emotion_net_finetuned.h5")
print("✅ Fine-tuning complete! Model saved as 'ADEDIWURA_emotion_net_finetuned.h5'")
