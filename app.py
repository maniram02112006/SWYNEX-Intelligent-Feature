"""
SWYNEX Task 3 - Intelligent Feature Prototype
Graphic Design Style Classifier & AI Design Insight Assistant (Streamlit web app)

Reuses the trained Keras model from Task 1:
  https://github.com/maniram02112006/SWYNEX-AI-Graphic-Design-Classifier

Architecture and Preprocessing Facts:
  - Input size : 160 x 160 pixels, 3 color channels (RGB)
  - Pixel range: raw 0-255 values. MobileNetV2 preprocessing (x / 127.5 - 1)
                 is embedded INSIDE the model, so the app must NOT rescale by 255.
  - Output     : 5 softmax probabilities
  - Class order: ["cinematic", "minimalist", "typography", "sports", "vintage"]

Task 3 Intelligent Feature:
  - AI Design Insight Assistant: Deterministic, rule-based design rationale,
    uncertainty and ambiguity detection (margin check), and actionable design
    refinement suggestions.
"""

from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import insights

# ----------------------------------------------------------------------
# Settings (all taken from the Task 1 training script)
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model" / "graphic_design_classifier.keras"
EXAMPLES_DIR = Path(__file__).parent / "examples"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = insights.CLASS_NAMES

ALLOWED_TYPES = ["jpg", "jpeg", "png"]
MAX_FILE_MB = 10


# ----------------------------------------------------------------------
# Model loading (cached so it loads only once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained classifier model...")
def load_classifier():
    """Load the Task 1 Keras model. Returns (model, error_message)."""
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found at: {MODEL_PATH}\n\n"
            "Download 'graphic_design_classifier.keras' from the Task 1 "
            "repository and place it inside the 'model' folder."
        )
    try:
        import keras  # imported here so a missing install gives a clear message

        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as error:  # noqa: BLE001 - show any loading problem
        return None, f"The model file could not be loaded: {error}"

    # Safety check: the model must match the class list in this app.
    output_units = model.output_shape[-1]
    if output_units != len(CLASS_NAMES):
        return None, (
            f"The model has {output_units} outputs but this app defines "
            f"{len(CLASS_NAMES)} classes. Check CLASS_NAMES in app.py."
        )
    return model, None


# ----------------------------------------------------------------------
# Image handling & validation
# ----------------------------------------------------------------------
def open_image(uploaded_file):
    """Validate and open the uploaded file. Returns (image, error_message)."""
    if uploaded_file is None:
        return None, "No file was uploaded."

    size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return None, f"The file is {size_mb:.1f} MB. Please upload an image under {MAX_FILE_MB} MB."

    try:
        # verify() checks the file is a genuine readable image; it must be re-opened after.
        Image.open(uploaded_file).verify()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None, "This file is not a valid or readable JPG/PNG image."

    if image.width < 32 or image.height < 32:
        return None, f"The image is too small ({image.width} x {image.height} px). Minimum required size is 32 x 32 px."

    return image, None


def preprocess(image: Image.Image) -> np.ndarray:
    """
    Match the Task 1 training pipeline:
      keras.utils.load_img(path, target_size=(160, 160))   -> RGB, nearest resize
      keras.utils.img_to_array(...)                        -> float32, 0-255
    The model itself applies MobileNetV2 preprocessing internally, so no /255 here.
    """
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize(IMAGE_SIZE, Image.NEAREST)
    array = np.asarray(image_resized, dtype="float32")
    return np.expand_dims(array, axis=0)  # shape: (1, 160, 160, 3)


def predict(model, image: Image.Image) -> np.ndarray:
    """Run model inference and validate outputs. Returns 1D probabilities array."""
    batch = preprocess(image)
    raw_predictions = model.predict(batch, verbose=0)

    if raw_predictions is None or len(raw_predictions) == 0:
        raise ValueError("Model returned an empty prediction.")

    probabilities = raw_predictions[0]

    # Validate predictions shape and values
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output probabilities, got {len(probabilities)}."
        )
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("Model generated non-finite (NaN/Inf) output values.")

    return probabilities


# ----------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Graphic Design Style Classifier",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 AI Graphic Design Style Classifier")
st.caption("SWYNEX Technologies Internship — Task 3: Intelligent Feature (AI Design Insight Assistant)")
st.write(
    "Upload a poster, banner, or artwork to predict its graphic design style and receive "
    "intelligent, rule-based design insights, ambiguity detection, and composition improvement tips."
)

model, load_error = load_classifier()
if load_error:
    st.error(load_error)
    st.stop()

input_mode = st.radio(
    "Choose image source:",
    ["Upload your own image", "Try an example from dataset"],
    horizontal=True,
)

image = None
image_caption = ""

if input_mode == "Upload your own image":
    uploaded = st.file_uploader("Choose an image (JPG or PNG, max 10 MB)", type=ALLOWED_TYPES)
    if uploaded is not None:
        image, image_error = open_image(uploaded)
        if image_error:
            st.error(image_error)
            image = None
        else:
            image_caption = uploaded.name
    else:
        st.info("Upload a JPG or PNG image to see design insights and style predictions.")
else:
    sample_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.exists() else []
    if not sample_files:
        st.warning("No sample images found in the 'examples' folder.")
    else:
        sample_names = [f.name for f in sample_files]
        chosen = st.selectbox("Select a sample image from the dataset:", sample_names)
        chosen_path = EXAMPLES_DIR / chosen
        try:
            image = Image.open(chosen_path)
            image.load()
            image_caption = f"Sample: {chosen}"
        except Exception as err:  # noqa: BLE001
            st.error(f"Could not load sample image: {err}")
            image = None

if image is not None:
    col_img, col_pred = st.columns([1, 1], gap="medium")

    with col_img:
        st.subheader("🖼️ Input Image")
        st.image(image, caption=image_caption, use_container_width=True)

    probabilities = None
    with col_pred:
        st.subheader("📊 Model Prediction")
        try:
            with st.spinner("Classifying design style..."):
                probabilities = predict(model, image)
        except Exception as error:  # noqa: BLE001
            st.error(f"Prediction failed: {error}")

        if probabilities is not None:
            try:
                analysis = insights.analyze_design(probabilities, CLASS_NAMES)
            except Exception as err:  # noqa: BLE001
                st.error(f"Analysis engine error: {err}")
                analysis = None

            if analysis is not None:
                predicted_style = analysis["predicted_class"].title()
                confidence_pct = analysis["confidence"] * 100

                st.markdown(f"### Predicted Style: **{predicted_style}**")
                st.metric("Top Confidence Score", f"{confidence_pct:.1f}%")

                # Uncertainty / Ambiguity Warning Banner
                if analysis["is_uncertain"]:
                    with st.container():
                        st.warning("⚠️ **Uncertainty Detected**")
                        for reason in analysis["uncertainty_reasons"]:
                            st.write(f"- {reason}")
                        st.info(
                            "💡 **Recommendation:** Review this design manually. It may blend multiple styles "
                            "or diverge from the training sample distribution."
                        )

                # Class breakdown
                st.markdown("#### Probability Breakdown Across Styles")
                for name, prob in analysis["ranked_classes"]:
                    st.write(f"**{name.title()}**: {prob * 100:.1f}%")
                    st.progress(float(min(max(prob, 0.0), 1.0)))

    # ------------------------------------------------------------------
    # Task 3 Intelligent Feature: AI Design Insight Assistant
    # ------------------------------------------------------------------
    if probabilities is not None and analysis is not None:
        st.divider()
        st.subheader("💡 AI Design Insights Assistant")

        insight_col1, insight_col2 = st.columns([1, 1], gap="medium")

        with insight_col1:
            st.markdown(f"#### 🔍 Visual Rationale: {analysis['style_title']}")
            st.write(f"*{analysis['style_summary']}*")
            st.write(analysis["visual_rationale"])

            if analysis["hybrid_blend_note"]:
                st.info(analysis["hybrid_blend_note"])

        with insight_col2:
            st.markdown("#### 🛠️ Design Improvement Suggestions")
            for idx, suggestion in enumerate(analysis["suggestions"], start=1):
                st.markdown(f"**{idx}.** {suggestion}")

        st.caption(f"📌 **Interpretation Notice:** {analysis['disclaimer']}")

# ----------------------------------------------------------------------
# Evaluation, Methodology & Limitations
# ----------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Evaluation, Methodology & Model Limitations (Task 3 Technical Notes)"):
    st.markdown(
        """
### Evaluation & System Architecture
- **Model Backbone:** Transfer learning using `MobileNetV2` with frozen feature extractors, GlobalAveragePooling2D, Dropout (0.2), and a 5-unit Dense softmax output layer.
- **Input Specifications:** 160 x 160 x 3 RGB image. Built-in MobileNetV2 normalizer scales `0–255` raw values to `[-1, 1]`.
- **Training Constraints:** The model was trained in Task 1 on a small illustrative dataset (20 AI-generated images, 4 per class).
- **Evaluation Status:** No un-seen benchmark test set was part of Task 1. As a result, no generalized accuracy metric is claimed. All performance evaluations in `evaluation/evaluation_report.md` reflect verified runs on existing samples.
- **Intelligent Feature:** The AI Design Insight Assistant uses deterministic, transparent heuristic rules derived from graphic design theory to explain predictions, identify multi-style ambiguity, and suggest actionable refinements without external API dependencies.
"""
    )
