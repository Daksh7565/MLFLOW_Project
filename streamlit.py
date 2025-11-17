import streamlit as st
import mlflow
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Disease Predictor",
    page_icon="🌿",
    layout="centered",
)

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "hybrid_image_classifier"
MODEL_STAGE = "None"

# --- Class Labels ---
CLASS_NAMES = [
    'Aphids',
    'Army worm',
    'Bacterial Blight',
    'Healthy',
    'Powdery Mildew',
    'Target spot'
]

# --- Model Loading ---
@st.cache_resource
def load_model_from_mlflow(model_name, model_stage):
    """Loads a model from the MLflow Model Registry."""
    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        st.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure the MLflow tracking server is running and the model name and stage are correct.")
        return None

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image to match the model's input requirements."""
    image = image.resize((180, 180))
    image_array = np.array(image)
    image_array = image_array / 255.0
    
    # --- START OF CORRECTION ---
    # Expand dimensions AND explicitly cast the array to float32
    processed_image = np.expand_dims(image_array, axis=0).astype(np.float32)
    # --- END OF CORRECTION ---
    
    return processed_image

# --- Main Application ---
def main():
    st.title("🌿 Crop Disease Classification")
    st.markdown("Upload an image of a crop leaf, and the AI will predict its condition.")

    model = load_model_from_mlflow(MODEL_NAME, MODEL_STAGE)

    if model:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Classify Image", type="primary"):
                with st.spinner("🧠 Analyzing the image..."):
                    processed_image = preprocess_image(image)
                    try:
                        prediction = model.predict(processed_image)
                        predicted_class_index = np.argmax(prediction)
                        predicted_class_name = CLASS_NAMES[predicted_class_index]
                        confidence = np.max(prediction)

                        st.success(f"**Prediction:** {predicted_class_name}")
                        st.info(f"**Confidence:** {confidence:.2%}")

                        st.subheader("Prediction Probabilities")
                        prob_per_class = {CLASS_NAMES[i]: prediction[0][i] for i in range(len(CLASS_NAMES))}
                        st.bar_chart(prob_per_class)

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()