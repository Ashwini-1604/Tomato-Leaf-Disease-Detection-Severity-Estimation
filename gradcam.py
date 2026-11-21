"""
gradcam.py
----------
Grad-CAM explainability for the tomato leaf disease classifier.

Produces a heatmap showing which pixels the model "looked at" most when
making its prediction (red/yellow = high attention, blue = low attention).

Works with the nested-model architecture produced by train.py:
    Input -> preprocess_input (Lambda) -> EfficientNetB0 (as a sub-model)
          -> GlobalAveragePooling2D -> Dropout -> Dense(128) -> Dropout -> Dense(softmax)

HOW TO RUN STANDALONE (on Colab or locally, after training):
    python gradcam.py --image_path sample_leaf.jpg --model_path tomato_model.h5

Saves: gradcam_output.png (original leaf + heatmap overlay side by side)

HOW IT'S USED INSIDE app.py (already wired in, no extra work needed):
    from gradcam import get_last_conv_layer_name, make_gradcam_heatmap, overlay_heatmap
    base_name, conv_name = get_last_conv_layer_name(model)
    heatmap, _ = make_gradcam_heatmap(input_arr, model, base_name, conv_name, pred_index=top_idx)
    gradcam_bgr = overlay_heatmap(bgr_image, heatmap)
"""

import argparse

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)


def get_last_conv_layer_name(model: tf.keras.Model):
    """
    Finds the nested base model (EfficientNetB0) inside the outer classification
    model, and the name of its last convolutional layer.

    Returns:
        (base_model_name, last_conv_layer_name)
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        # Model wasn't nested (e.g. base layers were merged into the outer
        # model directly) -- fall back to searching the outer model itself.
        base_model = model

    last_conv_name = None
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name

    if last_conv_name is None:
        raise ValueError(
            "Could not find a Conv2D layer inside the base model. "
            "Grad-CAM requires a convolutional layer to attach to."
        )

    return base_model.name, last_conv_name


def make_gradcam_heatmap(img_array, model, base_model_name, last_conv_layer_name, pred_index=None):
    """
    Args:
        img_array: preprocessed input, shape (1, 224, 224, 3), RAW pixel values
                   0-255 (this function applies EfficientNet preprocessing itself,
                   matching what train.py/app.py do internally).
        model: the full trained Keras model.
        base_model_name: name of the nested EfficientNetB0 layer.
        last_conv_layer_name: name of the last conv layer inside that base model.
        pred_index: which class to explain. Defaults to the model's top prediction.

    Returns:
        (heatmap, preds) -- heatmap is a 2D numpy array normalized to [0, 1];
        preds is the raw softmax output.
    """
    base_model = model.get_layer(base_model_name)

    # Sub-model: base model's input -> (last conv activations, base model output)
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )

    # Rebuild the "classifier head" (every layer that comes after the base
    # model in the outer model) as its own small model, so we can run the
    # base model's output back through it to get final class probabilities.
    base_index = model.layers.index(base_model)
    head_layers = model.layers[base_index + 1:]
    head_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = head_input
    for layer in head_layers:
        x = layer(x)
    classifier_model = tf.keras.Model(head_input, x)

    # The outer model applies EfficientNet's preprocess_input between its own
    # Input and the base model -- replicate that here since we're feeding the
    # base model directly.
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(
        tf.cast(img_array, tf.float32)
    )

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(preprocessed)
        tape.watch(conv_output)
        preds = classifier_model(base_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), preds.numpy()


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resizes the heatmap to the original image size and blends it on top
    using a JET colormap (red/yellow = high attention, blue = low)."""
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def _preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="tomato_model.h5")
    args = parser.parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(args.model_path)

    pil_image = Image.open(args.image_path)
    input_arr = _preprocess_for_model(pil_image)

    rgb_image = np.array(pil_image.convert("RGB").resize(IMG_SIZE))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    base_name, conv_name = get_last_conv_layer_name(model)
    print(f"Using base model '{base_name}', last conv layer '{conv_name}'")

    heatmap, preds = make_gradcam_heatmap(input_arr, model, base_name, conv_name)
    top_idx = int(np.argmax(preds[0]))
    print(f"Predicted class index {top_idx}, confidence {preds[0][top_idx] * 100:.2f}%")

    gradcam_bgr = overlay_heatmap(bgr_image, heatmap)

    # Side-by-side: original | heatmap overlay
    combined = np.hstack([bgr_image, gradcam_bgr])
    cv2.imwrite("gradcam_output.png", combined)
    print("Saved gradcam_output.png")


if __name__ == "__main__":
    main()
