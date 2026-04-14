import tensorflow as tf
from src.data_loader import get_data_loaders
import os
import json

print("Loading datasets...")
train_data, val_data = get_data_loaders(data_dir="data", batch_size=32, img_size=(256, 256))

print("Loading Phase 1 Model...")
model = tf.keras.models.load_model("models/model.h5")

print(f"Model successfully loaded! Total layers found: {len(model.layers)}")

# 1. UNFREEZE EVERYTHING FIRST
for layer in model.layers:
    layer.trainable = True

# 2. FREEZE THE BOTTOM 100 LAYERS
# This keeps the foundational edge-detection locked, but allows the top ~50 layers 
# to learn the highly specific biological textures of lung diseases.
print("Freezing the first 100 layers...")
for layer in model.layers[:100]:
    layer.trainable = False

# 3. RECOMPILE WITH A MICROSCOPIC LEARNING RATE
# 1e-5 is a tiny learning rate so we don't destroy the pre-trained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting Fine-Tuning (Phase 2)...")
# We only need about 10 epochs for fine-tuning
history = model.fit(train_data, validation_data=val_data, epochs=10)

print("Saving Fine-Tuned Model...")
model.save("models/model_finetuned.h5")

# Save history for comparison
os.makedirs("outputs/comparisons", exist_ok=True)
with open("outputs/comparisons/phase2.json", "w") as f:
    json.dump(history.history, f)

print("🎉 Fine-Tuning Complete!")