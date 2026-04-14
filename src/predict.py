import cv2
import numpy as np
from src.gradcam import get_gradcam, create_gradcam_visualization
import os

classes = ["Covid-19", "Emphysema", "Normal", "Pneumonia-Bacterial", "Pneumonia-Viral", "Tuberculosis"]

def predict_image(model, image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = cv2.resize(img, (256,256))
    
    # 🔥 THE FIX: Removed the / 255.0! We now pass the raw 0-255 pixels just like training.
    # We just convert it to float32 because the AI mathematically expects decimals.
    processed = np.float32(original)
    processed = processed.reshape(1,256,256,3)

    preds = model.predict(processed)[0]
    class_idx = np.argmax(preds)

    heatmap = get_gradcam(model, processed)
    overlay = create_gradcam_visualization(heatmap, original)

    return {
        "prediction": classes[class_idx],
        "confidence": float(preds[class_idx]),
        "all_probs": preds.tolist(),
        "heatmap": "outputs/heatmaps/overlay.jpg",
        "original": "outputs/heatmaps/original.jpg"
    }