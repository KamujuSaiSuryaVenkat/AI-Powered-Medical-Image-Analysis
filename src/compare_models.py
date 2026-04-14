import matplotlib.pyplot as plt
import json
import os

def save_history(history, filename):
    os.makedirs("outputs/comparisons", exist_ok=True)
    with open(f"outputs/comparisons/{filename}.json", "w") as f:
        json.dump(history.history, f)

def plot_comparison(file1, file2):
    with open(file1) as f:
        h1 = json.load(f)
    with open(file2) as f:
        h2 = json.load(f)

    plt.figure()
    plt.plot(h1['val_accuracy'], label='Before Fine-Tuning')
    plt.plot(h2['val_accuracy'], label='After Fine-Tuning')
    plt.legend()
    plt.title("Model Comparison")
    plt.savefig("outputs/comparisons/comparison.png")