"""
app.py (v2 — upgraded UI)
--------------------------
A polished, multi-page Streamlit app for the Tomato Leaf Disease Detector:
- Detect tab: upload -> classification + confidence gauge + severity + Grad-CAM + treatment advice
- Dashboard tab: charts summarizing everything you've scanned this session
- History tab: table of past scans (session-based)
- About tab: project info for your demo/report

Run with:
    streamlit run app.py

Required files in the same folder:
    - tomato_model.h5, class_names.json  (from train.py)
    - severity_estimation.py, gradcam.py (already in this project)
"""

import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from severity_estimation import estimate_severity, get_damage_overlay
from gradcam import get_last_conv_layer_name, make_gradcam_heatmap, overlay_heatmap

IMG_SIZE = (224, 224)

TREATMENT_ADVICE = {
    "Tomato___Bacterial_spot": "Use copper-based bactericides; avoid overhead watering; remove infected leaves.",
    "Tomato___Early_blight": "Apply fungicide (chlorothalonil/mancozeb); rotate crops; remove lower infected leaves.",
    "Tomato___Late_blight": "Act fast — apply fungicide immediately; destroy infected plants to stop spread.",
    "Tomato___Leaf_Mold": "Improve air circulation and reduce humidity; apply fungicide if severe.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves; apply fungicide; avoid wetting foliage.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Spray with miticide or neem oil; increase humidity around plants.",
    "Tomato___Target_Spot": "Apply fungicide; improve spacing for airflow; remove debris around plants.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies (vector); remove and destroy infected plants.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants; disinfect tools; avoid tobacco contact (virus vector).",
    "Tomato___healthy": "No action needed — plant looks healthy!",
}

SEVERITY_COLORS = {
    "Healthy / negligible damage": "#2ecc71",
    "Mild": "#f1c40f",
    "Moderate": "#e67e22",
    "Severe": "#e74c3c",
    "No leaf detected": "#95a5a6",
}


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .main { background-color: #fafafa; }
        .stApp { font-family: 'Segoe UI', sans-serif; }

        .hero {
            background: linear-gradient(135deg, #2e7d32 0%, #c0392b 100%);
            padding: 28px 32px;
            border-radius: 16px;
            color: white;
            margin-bottom: 24px;
        }
        .hero h1 { margin: 0; font-size: 30px; }
        .hero p { margin: 6px 0 0 0; opacity: 0.9; }

        .metric-card {
            background: white;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border-left: 6px solid #2e7d32;
        }
        .metric-card h3 { margin: 0; font-size: 14px; color: #666; font-weight: 500; }
        .metric-card p { margin: 4px 0 0 0; font-size: 24px; font-weight: 700; color: #222; }

        .advice-box {
            background: #fff8e1;
            border-left: 6px solid #f39c12;
            padding: 16px 20px;
            border-radius: 10px;
            font-size: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, color: str = "#2e7d32"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color};">
            <h3>{label}</h3>
            <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Model loading + inference helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("tomato_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_for_model(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    return np.expand_dims(arr, axis=0)


def confidence_gauge(confidence: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2e7d32"},
                "steps": [
                    {"range": [0, 50], "color": "#ffe0e0"},
                    {"range": [50, 80], "color": "#fff3cd"},
                    {"range": [80, 100], "color": "#d4edda"},
                ],
            },
            title={"text": "Model Confidence"},
        )
    )
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_detect(model, class_names):
    st.markdown("### 🔍 Upload a Leaf Image")
    uploaded_file = st.file_uploader("Choose a tomato leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload a leaf photo to get started. Sample photos work well — plain background, leaf filling most of the frame.")
        return

    pil_image = Image.open(uploaded_file)
    show_gradcam = st.checkbox("Show Grad-CAM explainability heatmap", value=True)

    with st.spinner("Analyzing leaf..."):
        input_arr = preprocess_for_model(pil_image)
        preds = model.predict(input_arr, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        predicted_class = class_names[top_idx]
        confidence = float(preds[top_idx]) * 100

        rgb_image = np.array(pil_image.convert("RGB"))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        severity = estimate_severity(bgr_image)
        damage_overlay = cv2.cvtColor(get_damage_overlay(bgr_image), cv2.COLOR_BGR2RGB)

        gradcam_img = None
        if show_gradcam:
            try:
                base_name, conv_name = get_last_conv_layer_name(model)
                heatmap, _ = make_gradcam_heatmap(input_arr, model, base_name, conv_name, pred_index=top_idx)
                gradcam_bgr = overlay_heatmap(bgr_image, heatmap)
                gradcam_img = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                st.warning(f"Grad-CAM unavailable for this model layer structure: {e}")

    # Save to session history
    st.session_state.setdefault("history", [])
    st.session_state["history"].append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "disease": predicted_class.replace("Tomato___", "").replace("_", " "),
            "confidence": round(confidence, 1),
            "severity_percent": severity["severity_percent"],
            "severity_label": severity["severity_label"],
        }
    )

    st.markdown("---")
    st.markdown("### Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Predicted Disease", predicted_class.replace("Tomato___", "").replace("_", " "))
    with col2:
        metric_card("Confidence", f"{confidence:.1f}%", color="#f39c12")
    with col3:
        sev_color = SEVERITY_COLORS.get(severity["severity_label"], "#2e7d32")
        metric_card("Severity", f"{severity['severity_label']} ({severity['severity_percent']}%)", color=sev_color)

    st.markdown("###")
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.image(pil_image, caption="Original", use_column_width=True)
    with col_img2:
        st.image(damage_overlay, caption="Damage highlighted (red)", use_column_width=True)
    with col_img3:
        if gradcam_img is not None:
            st.image(gradcam_img, caption="Grad-CAM — model attention", use_column_width=True)
        else:
            st.empty()

    st.plotly_chart(confidence_gauge(confidence), use_column_width=True)

    st.markdown("### 💊 Recommended Action")
    st.markdown(
        f'<div class="advice-box">{TREATMENT_ADVICE.get(predicted_class, "No specific advice available.")}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("See full confidence breakdown across all classes"):
        df = pd.DataFrame(
            {
                "Class": [c.replace("Tomato___", "").replace("_", " ") for c in class_names],
                "Confidence (%)": (preds * 100).round(2),
            }
        ).sort_values("Confidence (%)", ascending=False)
        fig = px.bar(df, x="Confidence (%)", y="Class", orientation="h", color="Confidence (%)",
                     color_continuous_scale="Greens")
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_column_width=True)


def page_dashboard():
    st.markdown("### 📊 Session Dashboard")
    history = st.session_state.get("history", [])

    if not history:
        st.info("No scans yet this session. Go to the Detect tab and upload a leaf image first.")
        return

    df = pd.DataFrame(history)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Scans", str(len(df)))
    with col2:
        metric_card("Avg. Confidence", f"{df['confidence'].mean():.1f}%", color="#f39c12")
    with col3:
        healthy_pct = (df["disease"] == "healthy").mean() * 100
        metric_card("Healthy Scans", f"{healthy_pct:.0f}%", color="#2ecc71")

    st.markdown("###")
    c1, c2 = st.columns(2)
    with c1:
        disease_counts = df["disease"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Count"]
        fig1 = px.pie(disease_counts, names="Disease", values="Count", title="Disease Distribution",
                      color_discrete_sequence=px.colors.sequential.RdYlGn_r)
        st.plotly_chart(fig1, use_column_width=True)

    with c2:
        severity_counts = df["severity_label"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig2 = px.bar(severity_counts, x="Severity", y="Count", color="Severity",
                      color_discrete_map=SEVERITY_COLORS, title="Severity Breakdown")
        st.plotly_chart(fig2, use_column_width=True)

    st.markdown("#### Confidence Over Time")
    fig3 = px.line(df, y="confidence", markers=True, title="Model Confidence per Scan (this session)")
    fig3.update_layout(xaxis_title="Scan #", yaxis_title="Confidence (%)")
    st.plotly_chart(fig3, use_column_width=True)


def page_history():
    st.markdown("### 📜 Scan History")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No scans yet this session.")
        return
    df = pd.DataFrame(history)
    st.dataframe(df, use_column_width=True)
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()


def page_about():
    st.markdown("### ℹ️ About This Project")
    st.write(
        """
        **Tomato Leaf Disease Detector** classifies 10 tomato leaf conditions
        (9 diseases + healthy) from a photo, estimates damage severity, and
        explains its decision using Grad-CAM.

        **Pipeline:**
        - Classification: EfficientNetB0 (transfer learning, fine-tuned)
        - Severity estimation: HSV color segmentation (classical CV, no extra labels needed)
        - Explainability: Grad-CAM heatmaps over the last convolutional layer
        - Dashboard: session-level analytics across all scans

        Built as a final-year ML project — see the README for training and
        deployment instructions.
        """
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Tomato Leaf Disease Detector", page_icon="🍅", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🍅 Tomato Leaf Disease Detector</h1>
            <p>Upload a leaf photo to detect disease, estimate severity, and see what the model is looking at.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, class_names = load_model()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect", "📊 Dashboard", "📜 History", "ℹ️ About"])
    with tab1:
        page_detect(model, class_names)
    with tab2:
        page_dashboard()
    with tab3:
        page_history()
    with tab4:
        page_about()


if __name__ == "__main__":
    main()
