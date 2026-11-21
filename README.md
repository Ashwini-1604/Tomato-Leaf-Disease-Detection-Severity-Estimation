# Tomato Leaf Disease Detection + Severity Estimation

Upload a photo of a tomato leaf and get:

- **Disease class** (9 diseases + healthy) with a confidence score
- **Severity estimate**: % of leaf area damaged, labelled Mild / Moderate / Severe
- **Grad-CAM heatmap** showing which part of the leaf drove the prediction
- A **Streamlit** app with Detect, Dashboard, History and About tabs

## How it works

| Part | Approach | File |
|---|---|---|
| Classification | EfficientNetB0 transfer learning (frozen base, then fine-tuning at a low learning rate), 224x224 input, augmentation, early stopping | `train.py` |
| Severity | Classical computer vision: HSV colour segmentation. Damaged % = leaf pixels that are not healthy green. No severity labels needed | `severity_estimation.py` |
| Explainability | Grad-CAM on the last conv layer of the EfficientNet base | `gradcam.py` |
| Evaluation | Confusion matrix + per-class precision/recall/F1 | `evaluate.py` |
| App | Streamlit UI | `app.py` |

## Run it

The trained model and dataset are **not stored in this repo**. You train once, then run the app.

1. **Get the data**: download the Tomato folders from the Kaggle dataset
   [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   and place them as `dataset/train/<class>/` and `dataset/valid/<class>/`.
2. **Train** (a free Colab T4 GPU works):
   ```bash
   pip install -r requirements.txt
   python train.py --data_dir dataset --epochs 15 --fine_tune_epochs 5
   ```
   This writes `tomato_model.h5` and `class_names.json` next to `app.py`.
3. **Evaluate** (optional): `python evaluate.py --data_dir dataset --model_path tomato_model.h5`
4. **Launch the app**: `streamlit run app.py`

`app.py` exits with an error if `tomato_model.h5` or `class_names.json` is missing.
More detail is in [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## Known limitations

- The dataset is PlantVillage-style: leaves photographed on plain backgrounds. Accuracy on field photos with clutter, shadows and different lighting will be lower.
- Severity is a colour heuristic, not a trained model. Anything that is not healthy green (yellowing, soil, shadow) counts as damage, and it has not been validated against expert severity scores.
- Dashboard and history are per session and reset on page refresh (no database).
