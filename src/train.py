import tensorflow as tf
from src.model import build_model
from src.data_loader import get_data_loaders
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ✅ Create required folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "outputs", "graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

print("📂 Loading datasets...")
train_data, val_data = get_data_loaders(
    data_dir="data",
    batch_size=32,
    img_size=(256, 256)
)

# ✅ Extract labels properly
print("⚖️ Calculating class weights...")
labels = []

for x, y in train_data:
    labels.extend(np.argmax(y.numpy(), axis=1))

labels = np.array(labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# ✅ Build model
print("🧠 Building model...")
model = build_model(num_classes=2)

# ✅ Compile (important if not done inside build_model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Callbacks (VERY IMPORTANT)
checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
]

# ✅ Train model
print("🚀 Starting training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ✅ Save final model (IMPORTANT FIX)
final_model_path = os.path.join(MODEL_DIR, "model_finetuned.h5")

print("💾 Saving final model...")
model.save(final_model_path)

print(f"✅ Model saved at: {final_model_path}")

# ✅ Save graphs
print("📊 Saving training graphs...")

# Accuracy graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(GRAPH_DIR, "accuracy.png"))

# Loss graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(GRAPH_DIR, "loss.png"))

print("🎉 Training complete and everything saved!")