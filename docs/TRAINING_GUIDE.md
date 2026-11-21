# Tomato Leaf Disease Detection + Severity Estimation

A deployable ML project that classifies tomato leaf diseases from a photo and
estimates how severe the damage is — built for a final-year project.

---

## 1. Project Overview

**Input:** A photo of a tomato leaf
**Output:**
- Disease classification (10 classes: 9 diseases + healthy)
- Confidence score
- Severity estimate (% of leaf damaged + Mild/Moderate/Severe label)
- Treatment recommendation

**Tech stack:**
- Model: EfficientNetB0 (transfer learning) via TensorFlow/Keras
- Severity: classical CV (HSV color segmentation) via OpenCV — no extra labeled data needed
- Deployment: Streamlit web app

---

## 2. Step-by-Step Plan

### Step 1 — Get the dataset
1. Go to Kaggle and download **"New Plant Diseases Dataset"** (search that exact
   name — it's the most popular, cleaned version of PlantVillage, already split
   into train/valid folders).
   - URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. You only need the `Tomato___*` folders. Extract just those into this structure:
   ```
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
       (same 10 subfolders)
   ```

### Step 2 — Train the model on Google Colab (free GPU)
1. Go to https://colab.research.google.com → New Notebook
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `train.py` and your `dataset/` folder (or mount Google Drive — recommended
   for large datasets: zip the dataset, upload to Drive, then unzip in Colab).
4. In a Colab cell:
   ```python
   !pip install tensorflow
   !python train.py --data_dir dataset --epochs 15 --fine_tune_epochs 5
   ```
5. This produces two files: `tomato_model.h5` and `class_names.json`.
   Download both to your project folder — you'll need them for deployment.
   Training takes roughly 30–60 minutes on a T4 GPU depending on dataset size.

### Step 2b — Evaluate the model (for your report)
Run this on Colab right after training, while `tomato_model.h5`,
`class_names.json`, and `dataset/` are all still there:
```bash
python evaluate.py --data_dir dataset --model_path tomato_model.h5
```
This produces:
- `confusion_matrix.png` — visual confusion matrix across all 10 classes
- `classification_report.txt` — precision, recall, F1-score per class + overall accuracy

Download both — put the confusion matrix image directly in your report, and
use the per-class numbers to discuss which diseases the model confuses most
(e.g. Early Blight vs Late Blight tend to be visually similar — a great
discussion point for your evaluation section).

### Step 2c — Grad-CAM explainability visualization
Shows *where* the model is looking when it classifies a leaf — strong for
the "explainable AI" angle evaluators like to see.
```bash
python gradcam.py --image_path sample_leaf.jpg --model_path tomato_model.h5
```
This saves `gradcam_output.png`: the original leaf with a heatmap overlay
(red/yellow = high attention, blue = low attention). Run it on a few sample
images from each class and include 2-3 examples in your report — ideally one
where the model focuses correctly on the lesion, which is good evidence the
model learned real disease features rather than background artifacts.

You can also wire this directly into `app.py` for a live demo — see the
comment block at the bottom of `gradcam.py` for the few lines needed.

### Step 3 — Test locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens a browser at `localhost:8501`. `app.py` now has 4 tabs:
- **🔍 Detect** — upload a leaf, see classification + confidence gauge + severity +
  Grad-CAM heatmap side-by-side + treatment advice + full per-class confidence chart
- **📊 Dashboard** — live charts (disease distribution pie, severity breakdown bar,
  confidence-over-time line) built from everything scanned this session
- **📜 History** — table of every scan this session, with a clear button
- **ℹ️ About** — project summary, handy for demoing to evaluators

Note: the dashboard/history are session-based (reset on page refresh) since
there's no database — good enough for a live demo. If you want scans to
persist across sessions, the easiest upgrade is writing each scan to a small
SQLite file instead of `st.session_state`.

### Step 4 — Deploy (pick one)

**Option A — Streamlit Community Cloud (easiest)**
1. Push this whole folder to a GitHub repo (include `tomato_model.h5` — if it's
   over 100MB, use Git LFS, or host the model on Hugging Face Hub instead and
   download it at app startup).
2. Go to https://share.streamlit.io → "New app" → connect your GitHub repo →
   select `app.py` as the entry point → Deploy.
3. You get a public URL you can share/demo with.

**Option B — Hugging Face Spaces**
1. Create a new Space at https://huggingface.co/spaces → SDK: Streamlit
2. Upload the same files (`app.py`, `severity_estimation.py`,
   `requirements.txt`, `tomato_model.h5`, `class_names.json`)
3. It auto-builds and gives you a public URL.

### Step 5 — Write up your report
Include these sections, since evaluators will expect them:
- **Dataset & preprocessing**: size, class balance, augmentation used
- **Architecture**: why EfficientNetB0 + transfer learning (fast convergence,
  strong ImageNet features transfer well to leaf textures)
- **Severity estimation method**: explain the HSV segmentation approach and
  why it avoids needing severity-labeled data
- **Evaluation**: accuracy, confusion matrix, precision/recall per class
  (add a short eval script if asked — happy to help with this)
- **Limitations**: background clutter can confuse the leaf mask; performance
  drops on leaves photographed in poor lighting; only covers tomato, not
  other crops

---

## 3. Making It Stand Out (optional but recommended)

- **Confusion matrix + per-class metrics** in your report, not just overall accuracy
- **Grad-CAM visualization** — show *where* the model is looking when it
  makes a prediction (great for the "explainability" angle evaluators like)
- **Mobile-friendly**: Streamlit apps work fine on phone browsers — demo it
  live on your phone during evaluation
- **Compare 2 architectures** (e.g., EfficientNetB0 vs a plain CNN from
  scratch) and report which performs better and why — this alone makes the
  project feel like real research instead of a tutorial clone

---

## 4. File Overview

| File | Purpose |
|---|---|
| `train.py` | Trains the EfficientNetB0 classifier, saves `tomato_model.h5` + `class_names.json` |
| `evaluate.py` | Generates confusion matrix + per-class precision/recall/F1 for your report |
| `gradcam.py` | Grad-CAM heatmap visualization for model explainability |
| `severity_estimation.py` | HSV-based severity % calculation + damage visualization |
| `app.py` | Streamlit app tying classification + severity together |
| `requirements.txt` | Python dependencies |

---

## 5. Common Issues

- **Model file too large for GitHub**: use Git LFS, or host on Hugging Face
  Hub and load it via `huggingface_hub.hf_hub_download()` inside `app.py`.
- **Low accuracy**: check class balance in your dataset; make sure you're
  using the `valid/` split for validation, not accidentally training on it.
- **Severity % looks off on some leaves**: the HSV thresholds are tuned for
  typical PlantVillage-style photos (leaf on plain background); adjust the
  ranges in `severity_estimation.py` if your test photos have busy backgrounds.
