"""
train.py
---------
Trains a tomato leaf disease classifier using transfer learning (EfficientNetB0).

HOW TO RUN THIS (on Google Colab):
1. Upload this file to Colab, or `git clone` your repo.
2. Runtime -> Change runtime type -> GPU (T4).
3. Download the dataset (see README.md for the Kaggle steps) so that you have
   a folder structure like:

   dataset/
     train/
       Tomato___Bacterial_spot/
       Tomato___Early_blight/
       Tomato___Late_blight/
       Tomato___Leaf_Mold/
       Tomato___Septoria_leaf_spot/
       Tomato___Spider_mites_Two-spotted_spider_mite/
       Tomato___Target_Spot/
       Tomato___Tomato_Yellow_Leaf_Curl_Virus/
       Tomato___Tomato_mosaic_virus/
       Tomato___healthy/
     valid/
       (same subfolders)

4. Run: python train.py --data_dir dataset --epochs 15
5. This saves the trained model as `tomato_model.h5` and the class names as
   `class_names.json`. Download both — you'll need them for the deployment app.
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes: int) -> tf.keras.Model:
    """Builds an EfficientNetB0 transfer-learning model."""
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False  # freeze base for initial training

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def fine_tune(model, base_model, unfreeze_from: int = 100):
    """Unfreezes the top layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to dataset dir with train/ and valid/ subfolders")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
    )

    class_names = list(train_gen.class_indices.keys())
    print("Classes found:", class_names)

    with open("class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    model, base_model = build_model(num_classes=len(class_names))

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ModelCheckpoint("tomato_model.h5", monitor="val_accuracy", save_best_only=True),
    ]

    print("\n--- Phase 1: training the classification head ---")
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n--- Phase 2: fine-tuning the top layers of EfficientNet ---")
    model = fine_tune(model, base_model)
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    model.save("tomato_model.h5")
    print("\nSaved model to tomato_model.h5 and classes to class_names.json")

    val_loss, val_acc = model.evaluate(valid_gen)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
