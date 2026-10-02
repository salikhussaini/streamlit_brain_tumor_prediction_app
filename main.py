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
# GradCAM Implementation
# ========================
class GradCAM:
    """
    Gradient-weighted Class Activation Maps (GradCAM)
    
    Visualizes which regions of the input image were important for the
    model's prediction by showing the gradient of the predicted class
    with respect to the feature maps of a target layer.
    
    This provides interpretability - shows where the model "looked" to
    make its prediction.
    """
    
    def __init__(self, model, layer_name):
        """
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer to visualize (can be nested like 'base_model.Conv_1')
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = None
        self._build_grad_model()
    
    def _build_grad_model(self):
        """Build a model that returns both predictions and gradients"""
        # Try to find the target layer
        try:
            # First try direct access
            self.target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            # If not found, try nested access (e.g., 'mobilenetv2_1.00_224.Conv_1')
            try:
                parts = self.layer_name.split('.')
                layer = self.model.get_layer(parts[0])
                for part in parts[1:]:
                    if hasattr(layer, 'get_layer'):
                        layer = layer.get_layer(part)
                    else:
                        raise ValueError(f"Cannot access layer {self.layer_name}")
                self.target_layer = layer
            except Exception as e:
                raise ValueError(f"Cannot find layer {self.layer_name}: {e}")
        
        # Create a model that outputs predictions and target layer outputs
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.output, self.target_layer.output]
        )
    
    def compute_gradcam(self, img_array, pred_index=None):
        """
        Compute GradCAM heatmap
        
        Args:
            img_array: Input image array (224, 224, 3)
            pred_index: Index of the class to visualize (None = use predicted class)
        
        Returns:
            heatmap: GradCAM heatmap (224, 224)
        """
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Record gradients
        with tf.GradientTape() as tape:
            predictions, target_layer_output = self.grad_model(img_batch, training=False)
            
            # Use predicted class if not specified
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the class channel
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the class channel with respect to target layer
        grads = tape.gradient(class_channel, target_layer_output)
        
        # Global Average Pooling of gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Compute weighted activation map
        target_layer_output = target_layer_output[0]
        heatmap = target_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap, axis=-1)
        
        # Normalize heatmap to 0-1 range
        heatmap = tf.maximum(heatmap, 0)
        heatmap /= (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()


def find_conv_layers(model):
    """
    Find all convolutional layers in the model, including nested layers in transfer learning models
    
    Returns layer names with full paths for nested models (e.g., 'mobilenetv2_1.00_224.Conv_1')
    """
    conv_layers = []
    
    def _search_layers(parent_layer, parent_name=""):
        """Recursively search for Conv2D layers"""
        if isinstance(parent_layer, tf.keras.layers.Conv2D):
            # Add this Conv2D layer
            full_name = f"{parent_name}.{parent_layer.name}" if parent_name else parent_layer.name
            conv_layers.append(full_name)
        
        # Handle layers that contain other layers (models, sequential, functional, etc.)
        if hasattr(parent_layer, 'layers'):
            for sublayer in parent_layer.layers:
                # Build the full path for nested layers
                if parent_name:
                    new_parent_name = f"{parent_name}.{parent_layer.name}"
                else:
                    new_parent_name = parent_layer.name
                
                _search_layers(sublayer, new_parent_name)
    
    # Search through all top-level layers in the model
    for layer in model.layers:
        _search_layers(layer)
    
    return conv_layers


def generate_gradcam_visualization(gradcam_heatmap, original_image, alpha=0.4):
    """
    Generate GradCAM visualization with heatmap overlay
    
    Args:
        gradcam_heatmap: GradCAM heatmap (224, 224)
        original_image: Original input image (224, 224, 3) in 0-1 range
        alpha: Transparency of heatmap overlay
    
    Returns:
        PIL Image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(
        gradcam_heatmap, 
        (original_image.shape[1], original_image.shape[0])
    )
    
    # Normalize and convert to uint8
    heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap (Jet colormap: blue=low importance, red=high importance)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to uint8 if needed
    if original_image.dtype != np.uint8:
        img_uint8 = (original_image * 255).astype(np.uint8)
    else:
        img_uint8 = original_image
    
    # If grayscale, convert to RGB
    if len(img_uint8.shape) == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    
    # Blend images
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlay)


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
        # Use safe_mode=False to allow loading nested models (Sequential containing Functional MobileNetV2)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss': focal_loss},
            safe_mode=False,
            compile=False
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
    diagram_lines.append(f"INPUT ({IMG_HEIGHT}×{IMG_WIDTH}×3)")
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
def predict_brain_tumor(preprocessed_image, model, class_names):
    """Run prediction on an already preprocessed image.
    
    Args:
        preprocessed_image: Image array already preprocessed with CLAHE + normalization
        model: Trained model
        class_names: List of class names
    
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score as percentage
        scores: Array of probabilities for all classes
    """
    # Add batch dimension
    img_batch = np.expand_dims(preprocessed_image, axis=0)
    
    # Model outputs logits
    predictions = model.predict(img_batch, verbose=0)
    
    # Convert logits to probabilities
    scores = tf.nn.softmax(predictions[0]).numpy()
    
    # Predicted class
    predicted_index = int(np.argmax(scores))
    predicted_class = class_names[predicted_index]
    
    # Confidence
    confidence = float(scores[predicted_index] * 100)
    
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
        
        # GradCAM Settings
        st.header("🔍 GradCAM Settings")
        show_gradcam = st.checkbox(
            "Show GradCAM Visualization",
            value=True,
            help="Display which regions the model focuses on for predictions"
        )
        gradcam_alpha = st.slider(
            "Heatmap Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = transparent, 1 = opaque"
        )
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
        img_resized = image_to_predict.convert("RGB").resize(
            (IMG_WIDTH, IMG_HEIGHT)
        )
        img_array = np.array(img_resized, dtype=np.uint8)
        
        # Apply the exact training preprocessing (CLAHE + intensity normalization)
        img_preprocessed = preprocess_medical_image(img_array)
        
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
                    img_preprocessed, model, class_names
                )
            
            # Prediction box
            if predicted_class == 'yes':
                st.error(f"🔴 **Model Prediction: Tumor Detected**")
            else:
                st.success("🟢 **Model Prediction: No Tumor Detected**")
            
            # Model Probability
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )
            st.caption(
                "Softmax output from the model; this is not a clinically calibrated probability."
            )
            
            # Score breakdown
            st.subheader("📊 Score Breakdown")
            score_dict = {class_names[i]: float(all_scores[i]) * 100 
                         for i in range(len(class_names))}
            
            for class_name, score in score_dict.items():
                st.progress(score / 100, text=f"{class_name}: {score:.2f}%")
            
            # Show source
            st.caption(f"*Image source: {image_source}*" if image_source else "")
        
        # ========================
        # GradCAM Visualization
        # ========================
        if show_gradcam:
            st.markdown("---")
            st.subheader("🔍 Model Interpretability - GradCAM Visualization")
            
            with st.spinner("Generating GradCAM heatmap..."):
                try:
                    # Find convolutional layers (including nested layers in transfer learning)
                    conv_layers = find_conv_layers(model)
                    
                    if not conv_layers:
                        # Fallback: search for any layer with 'conv' in the name
                        conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
                    
                    if conv_layers:
                        # Use the last convolutional layer for GradCAM
                        # (captures high-level features without being too abstract)
                        last_conv_layer = conv_layers[-1]
                        
                        try:
                            # Initialize GradCAM
                            gradcam = GradCAM(model, last_conv_layer)
                            
                            # Compute GradCAM heatmap using preprocessed image
                            # Get predicted class index
                            predicted_index = np.argmax(all_scores)
                            heatmap = gradcam.compute_gradcam(img_preprocessed, pred_index=predicted_index)
                            
                            # Generate visualization with user-selected alpha
                            gradcam_img = generate_gradcam_visualization(heatmap, img_preprocessed, alpha=gradcam_alpha)
                            
                            # Display side-by-side: original and GradCAM
                            col_orig_grad, col_sep_grad, col_gradcam = st.columns([1, 0.1, 1])
                            
                            with col_orig_grad:
                                st.write("**Original Image**")
                                st.image(img_preprocessed, use_container_width=True)
                            
                            with col_sep_grad:
                                st.write("")
                            
                            with col_gradcam:
                                st.write("**GradCAM Heatmap**")
                                st.write(f"*(Layer: {last_conv_layer})*")
                                st.image(gradcam_img, use_container_width=True)
                            
                            st.info(
                                "**GradCAM Explanation:**\n\n"
                                "The heatmap shows which regions of the brain MRI were most important "
                                "for the model's prediction. "
                                "\n- **Red regions** = High importance for the prediction\n"
                                "- **Blue regions** = Low importance for the prediction\n\n"
                                "This helps interpret WHY the model made its prediction."
                            )
                        
                        except Exception as grad_error:
                            st.warning(f"Could not generate GradCAM for layer '{last_conv_layer}': {str(grad_error)}")
                            
                            # Try alternative: first conv layer
                            if len(conv_layers) > 1:
                                st.info("Attempting with earlier convolutional layer...")
                                try:
                                    first_conv_layer = conv_layers[0]
                                    gradcam = GradCAM(model, first_conv_layer)
                                    predicted_index = np.argmax(all_scores)
                                    heatmap = gradcam.compute_gradcam(img_preprocessed, pred_index=predicted_index)
                                    gradcam_img = generate_gradcam_visualization(heatmap, img_preprocessed, alpha=gradcam_alpha)
                                    
                                    col_orig_grad, col_sep_grad, col_gradcam = st.columns([1, 0.1, 1])
                                    with col_orig_grad:
                                        st.write("**Original Image**")
                                        st.image(img_preprocessed, use_container_width=True)
                                    with col_sep_grad:
                                        st.write("")
                                    with col_gradcam:
                                        st.write("**GradCAM Heatmap**")
                                        st.write(f"*(Layer: {first_conv_layer})*")
                                        st.image(gradcam_img, use_container_width=True)
                                except Exception as fallback_error:
                                    st.error(f"Could not generate GradCAM: {str(fallback_error)}")
                            else:
                                st.error("Could not generate GradCAM with available layers")
                    else:
                        st.warning("⚠️ No convolutional layers found in model")
                        st.info(
                            "**For Transfer Learning Models:**\n\n"
                            "If using MobileNetV2, EfficientNet, or similar pretrained models, "
                            "the convolutional layers might be nested inside a base model layer. "
                            "\n\n**To enable GradCAM:**\n"
                            "1. Ensure your model has Conv2D layers accessible at the model level\n"
                            "2. Try extracting the base model and wrapping it in a new Sequential model\n"
                            "3. Check that the model was compiled with `trainable=True` for base layers"
                        )
                
                except Exception as e:
                    st.error(f"GradCAM Error: {str(e)}")
                    st.caption("Debug: Please check the model architecture in the sidebar")
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
