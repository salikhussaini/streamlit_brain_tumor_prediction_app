# 🧠 Brain Tumor Prediction App

A Streamlit web application for detecting brain tumors in MRI images using a deep learning CNN model.

## ⚠️ Disclaimer

**This tool is for educational purposes only and should NOT be used for medical diagnosis. Always consult with a qualified medical professional for any medical concerns.**

## Features

- 🖼️ Upload brain MRI images for tumor detection
- 🎯 Real-time predictions with confidence scores
- 📊 Score breakdown visualization
- 🎨 Test with sample images from the dataset
- 🏗️ Model architecture visualization
- 🔧 Support for multiple trained models
- 📈 Image preprocessing with CLAHE enhancement

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Clone or download the project

```bash
cd streamlit_brain_tumor_prediction_app
```

### 2. Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Live App

Try the app online here: **[Brain Tumor Prediction App](https://brain-tumor-prediction-app.streamlit.app/)**

### Running Locally

```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

### Making Predictions

1. **Upload an Image:**
   - Click "Upload Image" tab
   - Select a brain MRI image (JPG, PNG, GIF)
   - View the prediction result

2. **Use Test Samples:**
   - Click "Select Test Sample" tab
   - Choose from available sample images
   - View the prediction and confidence score

## Project Structure

```
streamlit_brain_tumor_prediction_app/
├── main.py                              # Main Streamlit application
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore file
├── README.md                           # This file
├── saved_models/
│   └── brain_tumor_model_*.keras       # Trained model files
│       └── class_names.npy             # Class labels (yes/no)
└── input/
    └── test/
        ├── yes/                        # Sample images with tumors
        └── no/                         # Sample images without tumors
```

## Model Information

- **Architecture:** Convolutional Neural Network (CNN) based on MobileNetV2
- **Input Size:** 224×224 pixels (RGB)
- **Output Classes:** 2 (yes = tumor, no = no tumor)
- **Loss Function:** Focal Loss (optimized for hard-to-detect tumors)
- **Preprocessing:** CLAHE + Intensity Normalization

### Image Preprocessing Steps

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   - Enhances local contrast in medical images
   - Makes subtle tumor features more visible

2. **Intensity Normalization:**
   - Standardizes pixel values
   - Improves model generalization across different MRI scanners

## Training the Model

If you need to train your own model, ensure you have the training script (`brain_tumor_prediction.py`) and run:

```bash
python brain_tumor_prediction.py
```

This will generate a timestamped model file in `saved_models/` directory.

## Dependencies

- **streamlit** - Web app framework
- **tensorflow** - Deep learning framework
- **numpy** - Numerical computing
- **pillow** - Image processing
- **opencv-python** - Computer vision library
- **scikit-learn** - Machine learning utilities

See `requirements.txt` for specific versions.

## Configuration

Key parameters in `main.py`:

```python
IMG_HEIGHT = 224  # Image height (must match training)
IMG_WIDTH = 224   # Image width (must match training)
TEST_DIR = CURRENT_FOLDER / 'input' / 'test'  # Test samples location
```

## Troubleshooting

### Model Not Found Error
- Ensure `saved_models/` directory exists
- Check that a trained model file is present
- Run the training script to generate a model

### Missing Sample Images
- Verify `input/test/yes/` and `input/test/no/` directories exist
- Add sample MRI images to these directories

### Out of Memory Error
- Reduce the number of images in memory
- Close other applications
- Use a machine with more available RAM

### TensorFlow Warnings
- These are normal and can be safely ignored
- Add `TF_CPP_MIN_LOG_LEVEL=2` environment variable to suppress them

**Windows:**
```bash
set TF_CPP_MIN_LOG_LEVEL=2
streamlit run main.py
```

**macOS/Linux:**
```bash
TF_CPP_MIN_LOG_LEVEL=2 streamlit run main.py
```

## Performance Notes

- Model predictions are cached in Streamlit for efficiency
- First prediction may take a few seconds
- Subsequent predictions are nearly instant
- Image preprocessing adds ~100-200ms per image

## Future Improvements

- [ ] Support for batch predictions
- [ ] Model comparison feature
- [ ] Detailed prediction explanations (Grad-CAM)
- [ ] ROC curves and performance metrics
- [ ] Multi-format MRI support (.dcm, .nii)
- [ ] Real-time camera feed support

## License

This project is for educational purposes.

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure you're using Python 3.8 or higher

---

**Last Updated:** 2026-09-26
