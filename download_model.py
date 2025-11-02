import os

# Path to your manually placed model
MODEL_PATH = os.path.join("saved_models", "ADEDIWURA_emotion_net.h5")

def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists in 'saved_models/'. You’re ready to run the app!")
    else:
        print("❌ Model not found. Please place 'ADEDIWURA_emotion_net.h5' inside the 'saved_models/' folder.")

if __name__ == "__main__":
    download_model()
