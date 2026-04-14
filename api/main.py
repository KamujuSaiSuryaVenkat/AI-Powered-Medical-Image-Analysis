from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# ✅ Load model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model_finetuned.h5")

model = tf.keras.models.load_model(model_path)

class_names = ["class1", "class2"]  # update if needed

print("✅ Model loaded successfully!")

# ✅ Image preprocessing
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ API route
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)
    predictions = model.predict(img)[0]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # ✅ all probabilities
    all_probs = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    # 🔥 DUMMY HEATMAP (temporary fix)
    heatmap = [float(x) for x in predictions]

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "heatmap": heatmap   # ✅ REQUIRED
    }