from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image

# --------------------- Setup ---------------------
app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "saved_models/ADEDIWURA_emotion_net.h5"
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Database setup
DB_PATH = "database.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT,
              image_path TEXT,
              predicted_emotion TEXT,
              timestamp TEXT)''')
conn.commit()
conn.close()

# --------------------- Routes ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    img_path = None

    if request.method == "POST":
        username = request.form["username"]
        file = request.files["image"]

        if file:
            # Save uploaded image
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # ----------------- Preprocess image -----------------
            img = Image.open(img_path).convert('RGB')  # Keep 3 channels
            img = img.resize((48, 48))                 # Resize to 48x48
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Shape (1,48,48,3)

            # Predict emotion
            preds = model.predict(img_array)
            emotion = emotion_labels[np.argmax(preds)]

            # Save to database
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, image_path, predicted_emotion, timestamp) VALUES (?, ?, ?, ?)",
                (username, img_path, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()

    return render_template("index.html", emotion=emotion, image=img_path)


# --------------------- Run App ---------------------
if __name__ == "__main__":
    app.run(debug=True)
