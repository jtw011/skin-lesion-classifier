# Skin Lesion Classification Prototype
A deep learning tool built with TensorFlow and EfficientNet to classify dermoscopy images as benign or malignant.

## Disclaimer
⚠️ **Educational Use Only:** This model is a research prototype. It has not been clinically validated and should not be used for actual medical diagnosis.

## Tech Stack
- **Framework:** TensorFlow/Keras
- **UI:** Streamlit
- **Architecture:** EfficientNetB0 with fine-tuning
- **Loss Function:** Custom Focal Loss (to handle class imbalance)

## Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`