from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import sqlite3
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image

# --------------------- TensorFlow CPU optimization ---------------------
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# --------------------- Setup ---------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max upload

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model globally
MODEL_PATH = "saved_models/ADEDIWURA_emotion_net.h5"
model = load_model(MODEL_PATH)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Database setup
DB_PATH = "database.db"
def init_db():
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

init_db()

# --------------------- Routes ---------------------
@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    img_path = None

    if request.method == "POST":
        username = request.form.get("username", "Anonymous")
        file = request.files.get("image")

        if file:
            # Save uploaded image safely
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            try:
                # ----------------- Preprocess image -----------------
                img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
                img = img.resize((48, 48))                  # Resize to 48x48
                img_array = np.array(img) / 255.0           # Normalize to 0-1
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

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

            except Exception as e:
                print(f"Error processing image: {e}")
                emotion = "Error processing image"

    return render_template("index.html", emotion=emotion, image=img_path)

# --------------------- Error Handlers ---------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    return "File is too large. Max size is 2 MB.", 413

# --------------------- Run App ---------------------
if __name__ == "__main__":
    # Render uses gunicorn; this is mainly for local testing
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
