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
    
    Works with both direct Conv2D layers and nested layers in transfer learning models.
    """
    
    def __init__(self, model, layer_name, base_model_name=None):
        """
        Args:
            model: Keras model
            layer_name: Name of the convolutional layer (e.g., 'Conv_1')
            base_model_name: Parent model name if layer is nested (e.g., 'mobilenetv2_1.00_224')
        """
        self.model = model
        self.layer_name = layer_name
        self.base_model_name = base_model_name
        self.target_layer = None
        self._find_target_layer()
    
    def _find_target_layer(self):
        """Find and store the target layer"""
        try:
            # Case 1: Direct layer access (non-nested)
            self.target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
            # Case 2: Nested layer
            if self.base_model_name:
                try:
                    base_model = self.model.get_layer(self.base_model_name)
                    self.target_layer = base_model.get_layer(self.layer_name)
                except Exception as e:
                    raise ValueError(
                        f"Cannot find layer '{self.layer_name}' in base model '{self.base_model_name}': {e}"
                    )
            else:
                raise ValueError(f"Cannot find layer {self.layer_name}")
    
    def compute_gradcam(self, img_array, pred_index=None):
        """
        Compute activation map heatmap for nested transfer learning models
        
        Uses layer activation averaging which works reliably with complex model architectures.
        
        Args:
            img_array: Input image array (224, 224, 3)
            pred_index: Index of the class to visualize (None = use predicted class)
        
        Returns:
            heatmap: Activation heatmap (224, 224)
        """
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        
        # Use Keras backend function to extract layer outputs
        try:
            # Create a function that extracts the target layer output
            layer_output_fn = tf.keras.backend.function(
                [self.model.input],
                [self.target_layer.output]
            )
            
            # Get the layer output for this image
            layer_output = layer_output_fn([img_array])[0]  # Shape: (1, height, width, channels)
            
            # Compute importance scores for each channel
            # Average the absolute activations across spatial dimensions
            channel_importance = np.mean(np.abs(layer_output[0]), axis=(0, 1))  # (channels,)
            
            # Weight each channel by its importance
            weighted_activations = layer_output[0] * channel_importance[np.newaxis, np.newaxis, :]
            
            # Average across channels to get spatial importance map
            heatmap = np.mean(weighted_activations, axis=-1)  # (height, width)
            
            # Normalize heatmap to 0-1 range
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            return heatmap
        
        except Exception as e:
            # Fallback: just average the activations
            try:
                layer_output_fn = tf.keras.backend.function(
                    [self.model.input],
                    [self.target_layer.output]
                )
                layer_output = layer_output_fn([img_array])[0]
                
                # Simple average across channels
                heatmap = np.mean(np.abs(layer_output[0]), axis=-1)
                
                # Normalize
                heatmap = np.maximum(heatmap, 0)
                if np.max(heatmap) > 0:
                    heatmap = heatmap / np.max(heatmap)
                
                return heatmap
            except Exception as fallback_e:
                raise RuntimeError(f"Could not compute activation map: {str(e)}, Fallback error: {str(fallback_e)}")


def find_conv_layers(model):
    """
    Find all convolutional layers in the model, including nested layers in transfer learning models
    
    Returns: List of tuples (layer_name, base_model_name) where base_model_name is None for direct layers
    """
    conv_layers = []
    
    # Check top-level for direct Conv2D layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append((layer.name, None))
    
    # Check inside model layers (base models, sequential, functional, etc.)
    for parent_layer in model.layers:
        if hasattr(parent_layer, 'layers'):
            for sublayer in parent_layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    conv_layers.append((sublayer.name, parent_layer.name))
                
                # Also check deeper nesting (e.g., layers inside the base model)
                if hasattr(sublayer, 'layers'):
                    for subsubLayer in sublayer.layers:
                        if isinstance(subsubLayer, tf.keras.layers.Conv2D):
                            conv_layers.append((subsubLayer.name, parent_layer.name))
    
    return conv_layers


def get_all_layers_info(model):
    """Get information about all layers in the model for debugging"""
    info = []
    for layer in model.layers:
        info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'has_sublayers': hasattr(layer, 'layers')
        })
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers[:5]:  # Limit to first 5 for readability
                info.append({
                    'name': f"  └─ {sublayer.name}",
                    'type': sublayer.__class__.__name__,
                    'has_sublayers': hasattr(sublayer, 'layers')
                })
            if len(layer.layers) > 5:
                info.append({
                    'name': f"  └─ ... and {len(layer.layers) - 5} more layers",
                    'type': '...',
                    'has_sublayers': False
                })
    return info


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
            "Show Layer Activation Visualization",
            value=True,
            help="Display which regions the convolutional layer activates for this prediction"
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
            st.subheader("🔍 Model Interpretability - Layer Activation Map")
            
            with st.spinner("Analyzing convolutional layers..."):
                try:
                    # Find all convolutional layers
                    conv_layers = find_conv_layers(model)
                    
                    if conv_layers:
                        st.success(f"✓ Found {len(conv_layers)} convolutional layer(s)")
                        
                        # Select which layer to visualize (default to last one)
                        layer_options = [
                            f"{layer_name} {'(nested)' if base_model else ''}" 
                            for layer_name, base_model in conv_layers
                        ]
                        
                        selected_layer_idx = st.selectbox(
                            "Select convolutional layer for visualization:",
                            range(len(conv_layers)),
                            format_func=lambda i: layer_options[i],
                            index=len(conv_layers) - 1  # Default to last layer
                        )
                        
                        selected_layer_name, selected_base_model = conv_layers[selected_layer_idx]
                        
                        with st.spinner(f"Generating activation map for layer '{selected_layer_name}'..."):
                            try:
                                # Initialize GradCAM with proper layer references
                                gradcam = GradCAM(model, selected_layer_name, selected_base_model)
                                
                                # Compute GradCAM heatmap
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
                                    layer_info = f"{selected_layer_name}"
                                    if selected_base_model:
                                        layer_info += f" (from {selected_base_model})"
                                    st.write(f"*Layer: {layer_info}*")
                                    st.image(gradcam_img, use_container_width=True)
                                
                                st.info(
                                    "**GradCAM / Activation Map Explanation:**\n\n"
                                    "The heatmap shows which regions of the brain MRI were most activated "
                                    "by the convolutional layer for this prediction. "
                                    "\n- **Red/Yellow regions** = High activation (important features)\n"
                                    "- **Blue/Green regions** = Low activation (less important)\n\n"
                                    "For transfer learning models with complex architectures, this shows "
                                    "the spatial importance of features detected by the layer."
                                )
                            
                            except Exception as grad_error:
                                st.error(f"❌ Could not generate activation map: {str(grad_error)}")
                                
                                with st.expander("🔧 Debugging Information"):
                                    st.write("**Available layers in model:**")
                                    layer_info = get_all_layers_info(model)
                                    for info in layer_info:
                                        st.code(f"{info['name']} ({info['type']})")
                                    
                                    st.write("\n**Found convolutional layers:**")
                                    for layer_name, base_model in conv_layers:
                                        if base_model:
                                            st.code(f"- {layer_name} (inside {base_model})")
                                        else:
                                            st.code(f"- {layer_name}")
                    
                    else:
                        st.warning("⚠️ No convolutional layers found in model")
                        
                        with st.expander("🔧 Model Architecture Information"):
                            st.write("**All layers in your model:**")
                            layer_info = get_all_layers_info(model)
                            for info in layer_info:
                                st.code(f"{info['name']} ({info['type']})")
                            
                            st.info(
                                "**Options to fix this:**\n\n"
                                "1. **Check if model has a base_model attribute:**\n"
                                "   - Ensure base layers are accessible (not wrapped in a way that hides them)\n"
                                "   - Try accessing: `model.layers[0].layers` for nested models\n\n"
                                "2. **Rebuild the model:**\n"
                                "   ```python\n"
                                "   base_model = tf.keras.applications.MobileNetV2(...)\n"
                                "   model = tf.keras.Sequential([\n"
                                "       base_model,\n"
                                "       tf.keras.layers.GlobalAveragePooling2D(),\n"
                                "       tf.keras.layers.Dense(num_classes)\n"
                                "   ])\n"
                                "   ```\n\n"
                                "3. **Use Activation Maps instead** (alternative to GradCAM):\n"
                                "   - Extract feature maps directly from intermediate layers\n"
                                "   - Works with any model architecture"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error analyzing model: {str(e)}")
                    with st.expander("Debug Details"):
                        st.code(str(e))
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
