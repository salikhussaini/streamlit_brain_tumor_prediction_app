"""
Streamlit App for Brain Tumor Prediction
Uses the exported model for real-time predictions
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import random
from io import StringIO
import sys
import cv2

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Prediction",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================
# Configuration
# ========================
CURRENT_FOLDER = Path(__file__).parent.resolve()
MODELS_DIR = CURRENT_FOLDER / 'saved_models'
CLASS_NAMES_FILE = str(MODELS_DIR / 'class_names.npy')
IMG_HEIGHT = 224  # Must match brain_tumor_prediction.py (MobileNetV2 native size)
IMG_WIDTH = 224   # Must match brain_tumor_prediction.py (MobileNetV2 native size)
TEST_DIR = CURRENT_FOLDER / 'input' / 'test'


# ========================
# Loss Functions
# ========================
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for hard-to-detect tumors
    
    Focuses training on hard-to-classify samples (tumors that are difficult to detect)
    Reduces weight of easy examples and emphasizes hard ones
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions (logits)
        alpha: Weighting factor (0-1), default 0.25
        gamma: Focusing parameter (0-5+), default 2.0
               Higher gamma = more focus on hard examples
    
    Returns:
        Focal loss value
    """
    # Compute sparse categorical cross-entropy
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # Calculate probability of true class
    p_t = tf.exp(-ce)
    
    # Apply focal weighting: (1-p_t)^gamma
    # This down-weights easy examples and up-weights hard ones
    focal = alpha * (1 - p_t) ** gamma * ce
    
    return tf.reduce_mean(focal)


# ========================
# Load Model
# ========================
def get_all_models():
    """Get list of all available models sorted by date (latest first)"""
    model_files = sorted(MODELS_DIR.glob('brain_tumor_model_*.keras'), reverse=True)
    return [(str(mf), mf.name) for mf in model_files]


def find_latest_model():
    """Find the latest timestamped model"""
    model_files = sorted(MODELS_DIR.glob('brain_tumor_model_*.keras'), reverse=True)
    if model_files:
        return str(model_files[0]), model_files[0].name
    return None, None


@st.cache_resource
def load_model_cached(model_path):
    """Load the model from a specific path (cached to avoid reloading)"""
    try:
        # Pass custom_objects so TensorFlow can find focal_loss function
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss': focal_loss}
        )
        class_names = np.load(CLASS_NAMES_FILE, allow_pickle=True).tolist()
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model has been trained and saved using the training script.")
        return None, None


def load_model():
    """Find and load the latest model or user-selected model"""
    models = get_all_models()
    
    if not models:
        st.error("No trained models found in saved_models directory")
        st.info("Please run the training script first")
        return None, None
    
    # If only one model, load it automatically
    if len(models) == 1:
        model_path, model_name = models[0]
        model, class_names = load_model_cached(model_path)
        if model is not None:
            st.sidebar.success(f"✓ Loaded: {model_name}")
        return model, class_names
    
    # If multiple models, let user select
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        # Create options for selectbox
        model_options = [f"⭐ {name} (Latest)" if i == 0 else name for i, (_, name) in enumerate(models)]
        
        selected_idx = st.selectbox(
            "Choose a model:",
            range(len(models)),
            format_func=lambda i: model_options[i],
            help="Select which trained model to use for predictions"
        )
        
        selected_model_path, selected_model_name = models[selected_idx]
        
        # Load selected model
        model, class_names = load_model_cached(selected_model_path)
        if model is not None:
            st.success(f"✓ Loaded: {selected_model_name}")
        
        return model, class_names


def get_sample_images(limit_per_class=5):
    """Get list of available sample images from test directory"""
    if not TEST_DIR.exists():
        return {}
    
    samples = {}
    for class_dir in TEST_DIR.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if images:
                # Limit to specified number per class
                samples[class_dir.name] = sorted(images)[:limit_per_class]
    
    return samples


def get_model_summary(model):
    """Get model summary as a string"""
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    return summary_string.getvalue()


def preprocess_medical_image(image_array):
    """
    Apply CLAHE and intensity normalization for medical imaging
    
    CLAHE: Contrast Limited Adaptive Histogram Equalization
    - Enhances local contrast in medical images
    - Makes subtle tumor features visible
    
    Intensity Normalization:
    - Standardizes pixel intensity across different scanners
    - Improves model generalization
    """
    # Convert to uint8 if needed
    if image_array.dtype != np.uint8:
        # If already in 0-1 range, convert to 0-255
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Convert to grayscale if RGB (for CLAHE)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to float32 in range [0, 1]
    enhanced = enhanced.astype(np.float32) / 255.0
    
    # Intensity normalization (standardization)
    mean = enhanced.mean()
    std = enhanced.std() + 1e-7
    normalized = (enhanced - mean) / std
    
    # Clip to reasonable range
    normalized = np.clip(normalized, -2.0, 2.0)
    
    # Convert back to 0-1 range
    normalized = (normalized + 2.0) / 4.0
    
    # If original was RGB, convert back to RGB by repeating channel
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        normalized = np.stack([normalized, normalized, normalized], axis=-1)
    
    return normalized


def generate_model_diagram(model):
    """Generate a dynamic architecture diagram from model structure"""
    diagram_lines = []
    diagram_lines.append("INPUT (180×180×3)")
    diagram_lines.append("       ↓")
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.__class__.__name__
        layer_config = layer.get_config()
        
        # Get output shape safely
        try:
            if hasattr(layer, '_batch_input_shape') and layer._batch_input_shape:
                output_shape = layer._batch_input_shape
            else:
                output_shape = None
        except:
            output_shape = None
        
        shape_str = ""
        if output_shape:
            shape_str = "×".join(str(d) if d else "?" for d in output_shape[1:])
        
        # Create layer representation
        if "Conv2D" in layer_name:
            filters = layer_config.get('filters', '?')
            kernel = layer_config.get('kernel_size', [3, 3])
            diagram_lines.append(f"┌─────────────────────────────┐")
            diagram_lines.append(f"│ Conv2D({filters}, {kernel[0]}×{kernel[0]})      │")
            if shape_str:
                diagram_lines.append(f"│ Output: {shape_str:<19} │")
            diagram_lines.append(f"└─────────────────────────────┘")
            diagram_lines.append("       ↓")
        
        elif "BatchNormalization" in layer_name:
            diagram_lines.append(f"  → BatchNormalization")
        
        elif "MaxPooling2D" in layer_name:
            pool_size = layer_config.get('pool_size', [2, 2])
            diagram_lines.append(f"  → MaxPool({pool_size[0]}×{pool_size[0]})")
        
        elif "Dropout" in layer_name:
            rate = layer_config.get('rate', 0.5)
            diagram_lines.append(f"  → Dropout({rate})")
            diagram_lines.append("       ↓")
        
        elif "Flatten" in layer_name:
            diagram_lines.append(f"┌─────────────────────────────┐")
            diagram_lines.append(f"│ Flatten                     │")
            diagram_lines.append(f"└─────────────────────────────┘")
            diagram_lines.append("       ↓")
        
        elif "Dense" in layer_name:
            units = layer_config.get('units', '?')
            activation = layer_config.get('activation', 'linear')
            diagram_lines.append(f"┌─────────────────────────────┐")
            diagram_lines.append(f"│ Dense({units}, {activation:<11}) │")
            if shape_str:
                diagram_lines.append(f"│ Output: {shape_str:<19} │")
            diagram_lines.append(f"└─────────────────────────────┘")
            if i < len(model.layers) - 1:
                diagram_lines.append("       ↓")
    
    diagram_lines.append("")
    diagram_lines.append(f"Total Parameters: {model.count_params():,}")
    
    return "\n".join(diagram_lines)


# ========================
# Prediction Function
# ========================
def predict_brain_tumor(image, model, class_names):
    """Predict if image contains brain tumor"""
    # Ensure image is RGB (handle grayscale, RGBA, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Convert to numpy array with correct dtype
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Ensure correct shape (height, width, 3)
    if len(img_array.shape) != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] != 3:
        # If it has wrong number of channels, convert
        img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    scores = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(scores)]
    confidence = 100 * np.max(scores)
    
    return predicted_class, confidence, scores


# ========================
# UI Components
# ========================
def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🧠 Brain Tumor Prediction")
    st.markdown("---")
    st.write(
        "Upload a brain MRI image to detect the presence of a tumor. "
        "The model analyzes the image and provides a prediction with confidence score."
    )
    
    # Load model
    model, class_names = load_model()
    
    if model is None or class_names is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        
        st.write(f"**Input Size:** {IMG_HEIGHT}×{IMG_WIDTH}")
        st.write(f"**Classes:** {', '.join(class_names)}")
        st.markdown("---")
        
        # Model Architecture Summary
        with st.expander("🏗️ Model Architecture"):
            # Display dynamic architecture diagram
            st.subheader("Architecture Diagram")
            architecture_diagram = generate_model_diagram(model)
            st.code(architecture_diagram, language="text")
            
            st.markdown("---")
            
            # Display detailed summary
            st.subheader("Detailed Layer Summary")
            model_summary_text = get_model_summary(model)
            st.code(model_summary_text, language="text")
        
        st.markdown("---")
        
        st.header("📋 About")
        st.write(
            "This model is trained using a CNN (Convolutional Neural Network) "
            "to classify brain MRI images as either containing a tumor (Yes) or not (No)."
        )
        
        st.header("⚠️ Disclaimer")
        st.warning(
            "This tool is for educational purposes only and should not be used "
            "for medical diagnosis. Always consult with a qualified medical professional."
        )
    
    # Main content
    input_method = st.radio(
        "Choose how to provide an image:",
        options=["Upload Image", "Select Test Sample"],
        horizontal=True
    )
    
    st.markdown("---")
    
    image_to_predict = None
    image_source = None
    
    if input_method == "Upload Image":
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=["jpg", "jpeg", "png", "gif"],
            help="Upload a brain MRI image for prediction"
        )
        
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)
            image_source = "uploaded"
    
    else:  # Select Test Sample
        st.subheader("🎯 Test with Sample Images")
        samples = get_sample_images()
        
        if samples:
            # Create a list of all sample images with their class labels
            sample_options = []
            for class_name, images in sorted(samples.items()):
                for img_path in images:
                    sample_options.append((f"[{class_name.upper()}] {img_path.stem}", img_path))
            
            if sample_options:
                selected_sample = st.selectbox(
                    "Select a test image...",
                    options=range(len(sample_options)),
                    format_func=lambda i: sample_options[i][0],
                    help="Select from available test samples"
                )
                sample_image_path = sample_options[selected_sample][1]
                image_to_predict = Image.open(sample_image_path)
                image_source = "sample"
        else:
            st.info("No sample images found in input/test directory")
    
    # ========================
    # Image Processing & Prediction
    # ========================
    
    if image_to_predict is not None:
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Prepare image data
        img_resized = image_to_predict.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # Ensure correct shape
        if len(img_array.shape) != 3:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] != 3:
            img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)
        
        # Apply preprocessing
        img_preprocessed = preprocess_medical_image((img_array * 255).astype(np.uint8))
        
        # Display images side-by-side
        st.subheader("📸 Image Processing")
        col_original, col_separator, col_processed = st.columns([1, 0.1, 1])
        
        with col_original:
            st.write("**Original Image**")
            st.image(img_resized, use_container_width=True)
        
        with col_separator:
            st.write("")
        
        with col_processed:
            st.write("**Preprocessed Image** (CLAHE + Normalized)")
            # Convert preprocessed image back to PIL for display
            preprocessed_display = (img_preprocessed * 255).astype(np.uint8)
            if len(preprocessed_display.shape) == 3:
                preprocessed_pil = Image.fromarray(preprocessed_display)
            else:
                preprocessed_pil = Image.fromarray(preprocessed_display)
            st.image(preprocessed_pil, use_container_width=True)
        
        st.info(
            "**Processing Steps:**\n\n"
            "1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): "
            "Enhances local contrast to make tumor features more visible\n\n"
            "2. **Intensity Normalization**: Standardizes pixel values across different MRI scanners "
            "to improve model generalization"
        )
        
        st.markdown("---")
        st.subheader("🎯 Prediction")
        
        col_pred_left, col_pred_right = st.columns([1, 1])
        
        with col_pred_left:
            st.write("**Input Image**")
            st.image(image_to_predict, use_container_width=True)
        
        with col_pred_right:
            # Make prediction
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_scores = predict_brain_tumor(
                    image_to_predict, model, class_names
                )
            
            # Prediction box
            if predicted_class == 'yes':
                st.error(f"🔴 **Prediction: Tumor Detected**")
            else:
                st.success(f"🟢 **Prediction: No Tumor**")
            
            # Confidence
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )
            
            # Score breakdown
            st.subheader("📊 Score Breakdown")
            score_dict = {class_names[i]: float(all_scores[i].numpy()) * 100 
                         for i in range(len(class_names))}
            
            for class_name, score in score_dict.items():
                st.progress(score / 100, text=f"{class_name}: {score:.2f}%")
            
            # Show source
            st.caption(f"*Image source: {image_source}*" if image_source else "")
    else:
        st.info("👆 Upload an image or select a test sample to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>Brain Tumor Prediction Model | Powered by TensorFlow & Streamlit</small>
        </div>
        """,
        unsafe_allow_html=True
    )


# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    main()
