"""
evaluate.py
-----------
Generates evaluation metrics for your report: confusion matrix,
per-class precision/recall/F1, and overall accuracy.

HOW TO RUN (on Colab, after training):
    python evaluate.py --data_dir dataset --model_path tomato_model.h5

Requires: tomato_model.h5, class_names.json (both produced by train.py)
and the same dataset/valid/ folder used during training.

Outputs:
    - confusion_matrix.png
    - classification_report.txt (precision/recall/F1 per class)
    - prints overall accuracy to console
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset dir (must contain a valid/ subfolder)")
    parser.add_argument("--model_path", type=str, default="tomato_model.h5")
    parser.add_argument("--class_names_path", type=str, default="class_names.json")
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    with open(args.class_names_path) as f:
        class_names = json.load(f)
    short_names = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    valid_gen = valid_datagen.flow_from_directory(
        f"{args.data_dir}/valid",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # important: keep order so predictions line up with labels
    )

    print("Running predictions on validation set...")
    preds = model.predict(valid_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = valid_gen.classes

    # --- Overall accuracy ---
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall validation accuracy: {acc:.4f}\n")

    # --- Per-class precision/recall/F1 ---
    report = classification_report(y_true, y_pred, target_names=short_names, digits=4)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(f"Overall accuracy: {acc:.4f}\n\n")
        f.write(report)
    print("Saved classification_report.txt")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Tomato Leaf Disease Classifier")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
