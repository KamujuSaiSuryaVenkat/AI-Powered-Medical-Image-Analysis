import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model_path="models/model.h5", test_dir="data/test"):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test data...")
    # UPDATED: color_mode is now "rgb" to match MobileNetV2!
    # We must keep shuffle=False so the predictions match the correct labels
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(256, 256),
        color_mode="rgb", 
        batch_size=32,
        label_mode="categorical",
        shuffle=False 
    )
    
    class_names = test_data.class_names
    
    print("Evaluating model... this might take a minute.")
    # This is exactly the prediction logic you mentioned!
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    
    # Extract true labels
    y_true = []
    for images, labels in test_data.unbatch():
        y_true.append(np.argmax(labels.numpy()))
    y_true = np.array(y_true)
    
    # 1. Print Classification Report (Accuracy, Precision, Recall)
    print("\n" + "="*40)
    print("--- Classification Report ---")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Plot Confusion Matrix (The Seaborn code you provided)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    print("\n✅ Confusion matrix saved to outputs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()