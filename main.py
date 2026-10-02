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
# Visualization Alternatives
# ========================
class SaliencyMap:
    """Compute saliency maps using image gradients"""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, img_array, pred_index=None):
        """Compute gradient-based saliency map"""
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor, training=False)
            
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, img_tensor)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)
        saliency = saliency[0].numpy()
        
        # Normalize
        saliency = np.maximum(saliency, 0)
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
        
        return saliency


class GradientInput:
    """Compute Gradient × Input attribution (emphasizes gradients from important pixels)"""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, img_array, pred_index=None):
        """Compute gradient × input attribution map"""
        img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor, training=False)
            
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, img_tensor)
        grads = tf.abs(grads)[0].numpy()  # Remove batch dimension
        
        # Multiply by input (emphasize gradients from high-intensity pixels)
        attribution = grads * img_array
        
        # Take max across channels
        attribution_map = np.max(attribution, axis=-1)
        
        # Normalize
        attribution_map = np.maximum(attribution_map, 0)
        if np.max(attribution_map) > 0:
            attribution_map = attribution_map / np.max(attribution_map)
        
        return attribution_map


class OcclusionSensitivity:
    """Visualize model sensitivity by occluding image regions"""
    
    def __init__(self, model):
        self.model = model
    
    def compute(self, img_array, patch_size=16):
        """
        Compute occlusion sensitivity map
        Shows how much the prediction changes when different regions are masked
        """
        img_batch = np.expand_dims(img_array, 0)
        original_pred = self.model.predict(img_batch, verbose=0)
        original_score = np.max(original_pred[0])
        
        sensitivity_map = np.zeros((img_array.shape[0], img_array.shape[1]))
        
        # Slide a patch across the image
        for i in range(0, img_array.shape[0], patch_size):
            for j in range(0, img_array.shape[1], patch_size):
                # Create occluded image (black out the patch)
                occluded_img = img_array.copy()
                occluded_img[i:i+patch_size, j:j+patch_size] = 0
                
                # Get prediction
                occluded_batch = np.expand_dims(occluded_img, 0)
                occluded_pred = self.model.predict(occluded_batch, verbose=0)
                occluded_score = np.max(occluded_pred[0])
                
                # Sensitivity = how much prediction dropped
                sensitivity = original_score - occluded_score
                sensitivity_map[i:i+patch_size, j:j+patch_size] = sensitivity
        
        # Normalize
        sensitivity_map = np.maximum(sensitivity_map, 0)
        if np.max(sensitivity_map) > 0:
            sensitivity_map = sensitivity_map / np.max(sensitivity_map)
        
        return sensitivity_map


class FeatureMapVisualizer:
    """Visualize feature maps from intermediate layers"""
    
    def __init__(self, model, layer_name, base_model_name=None):
        self.model = model
        self.layer_name = layer_name
        self.base_model_name = base_model_name
    
    def compute(self, img_array):
        """Extract and visualize feature maps"""
        try:
            # Get the target layer
            if self.base_model_name:
                base_model = self.model.get_layer(self.base_model_name)
                target_layer = base_model.get_layer(self.layer_name)
            else:
                target_layer = self.model.get_layer(self.layer_name)
            
            # Create feature extraction model
            feature_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=target_layer.output
            )
            
            # Get features
            img_batch = np.expand_dims(img_array, 0)
            features = feature_model.predict(img_batch, verbose=0)
            
            # Average across channels to get a 2D map
            feature_map = np.mean(features[0], axis=-1)
            
            # Normalize
            feature_map = np.maximum(feature_map, 0)
            if np.max(feature_map) > 0:
                feature_map = feature_map / np.max(feature_map)
            
            return feature_map
        except Exception as e:
            raise ValueError(f"Could not extract feature map: {e}")


class GradCAM:
    """Original GradCAM Implementation (kept for reference)"""
    
    def __init__(self, model, layer_name, base_model_name=None):
        self.model = model
        self.layer_name = layer_name
        self.base_model_name = base_model_name
        self.target_layer = None
        self._find_target_layer()
    
    def _find_target_layer(self):
        """Find and store the target layer"""
        try:
            self.target_layer = self.model.get_layer(self.layer_name)
        except ValueError:
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
        """Fallback activation map computation"""
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Simple approach: average layer output
        try:
            # Build sub-model to get layer outputs
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.target_layer.output
            )
            layer_output = intermediate_model.predict(img_batch, verbose=0)
            
            # Average across channels
            heatmap = np.mean(layer_output[0], axis=-1)
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            return heatmap
        except:
            # Last resort: return zeros
            return np.zeros((img_array.shape[0], img_array.shape[1]))


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
        # Model Interpretability Visualizations
        # ========================
        if show_gradcam:
            st.markdown("---")
            st.subheader("🔍 Model Interpretability - Multiple Visualization Options")
            
            # Choose visualization method
            viz_method = st.radio(
                "Select visualization method:",
                options=[
                    "Saliency Map (Gradient-based)",
                    "Gradient × Input",
                    "Occlusion Sensitivity"
                ],
                horizontal=True,
                help="Different methods to understand model decisions"
            )
            
            with st.spinner("Generating visualization..."):
                try:
                    if viz_method == "Saliency Map (Gradient-based)":
                        st.info(
                            "**Saliency Map:** Shows which input pixels have the strongest gradient "
                            "with respect to the predicted class. "
                            "Bright regions = pixels most important for the prediction."
                        )
                        
                        saliency = SaliencyMap(model)
                        predicted_index = np.argmax(all_scores)
                        heatmap = saliency.compute(img_preprocessed, pred_index=predicted_index)
                        
                        gradcam_img = generate_gradcam_visualization(heatmap, img_preprocessed, alpha=gradcam_alpha)
                        
                        col_orig, col_sep, col_viz = st.columns([1, 0.1, 1])
                        with col_orig:
                            st.write("**Original Image**")
                            st.image(img_preprocessed, use_container_width=True)
                        with col_sep:
                            st.write("")
                        with col_viz:
                            st.write("**Saliency Map**")
                            st.image(gradcam_img, use_container_width=True)
                    
                    elif viz_method == "Gradient × Input":
                        st.info(
                            "**Gradient × Input:** Multiplies gradients by input intensities. "
                            "Emphasizes gradients from high-intensity regions (where image features are strong). "
                            "Bright regions = pixels with strong features AND high influence on prediction."
                        )
                        
                        grad_input = GradientInput(model)
                        predicted_index = np.argmax(all_scores)
                        heatmap = grad_input.compute(img_preprocessed, pred_index=predicted_index)
                        
                        gradcam_img = generate_gradcam_visualization(heatmap, img_preprocessed, alpha=gradcam_alpha)
                        
                        col_orig, col_sep, col_viz = st.columns([1, 0.1, 1])
                        with col_orig:
                            st.write("**Original Image**")
                            st.image(img_preprocessed, use_container_width=True)
                        with col_sep:
                            st.write("")
                        with col_viz:
                            st.write("**Gradient × Input**")
                            st.image(gradcam_img, use_container_width=True)
                    
                    else:  # Occlusion Sensitivity
                        st.info(
                            "**Occlusion Sensitivity:** Slides a patch across the image and shows "
                            "how much the prediction changes. Bright regions = important for the prediction."
                        )
                        
                        with st.spinner("Computing occlusion sensitivity (this may take a moment)..."):
                            occlusion = OcclusionSensitivity(model)
                            heatmap = occlusion.compute(img_preprocessed, patch_size=16)
                            
                            gradcam_img = generate_gradcam_visualization(heatmap, img_preprocessed, alpha=gradcam_alpha)
                            
                            col_orig, col_sep, col_viz = st.columns([1, 0.1, 1])
                            with col_orig:
                                st.write("**Original Image**")
                                st.image(img_preprocessed, use_container_width=True)
                            with col_sep:
                                st.write("")
                            with col_viz:
                                st.write("**Occlusion Sensitivity**")
                                st.image(gradcam_img, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Visualization failed: {str(e)}")
                    st.caption("Try the other visualization method")
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
