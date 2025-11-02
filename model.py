import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import urllib.request
import zipfile

# Download FER2013 dataset automatically if not exists
def download_dataset():
    if not os.path.exists("fer2013.zip"):
        print("📥 Downloading FER2013 dataset...")
        url = "https://github.com/muxspace/facial_expressions/raw/master/fer2013/fer2013.csv"
        urllib.request.urlretrieve(url, "fer2013.csv")
        print("✅ Download completed!")
    else:
        print("✅ Dataset already downloaded.")

def load_dataset():
    print("📄 Loading dataset...")
    data = pd.read_csv("fer2013.csv")
    pixels = data["pixels"].tolist()
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(" ")]
        face = np.asarray(face).reshape(48, 48, 1)
        faces.append(face)
    faces = np.asarray(faces).astype("float32") / 255.0
    emotions = to_categorical(data["emotion"], num_classes=7)
    return train_test_split(faces, emotions, test_size=0.2)

def build_model():
    print("🧠 Building CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    download_dataset()
    X_train, X_val, y_train, y_val = load_dataset()
    model = build_model()

    print("🎯 Training model, please wait...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/ADEDIWURA_emotion_net.h5")
    print("✅ Model saved successfully at saved_models/ADEDIWURA_emotion_net.h5")

if __name__ == "__main__":
    main()
