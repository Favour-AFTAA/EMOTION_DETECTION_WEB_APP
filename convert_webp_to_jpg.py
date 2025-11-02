import os
from PIL import Image

# Path to your dataset
DATASET_DIR = "emotion_dataset"

# Loop through all subfolders
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(".webp"):
            webp_path = os.path.join(root, file)
            jpg_path = os.path.join(root, file.rsplit(".", 1)[0] + ".jpg")

            # Open and convert to JPG
            with Image.open(webp_path) as img:
                rgb_img = img.convert("RGB")  # Convert to RGB
                rgb_img.save(jpg_path, "JPEG")

            print(f"Converted: {webp_path} → {jpg_path}")
