import os
import torch
import boto3
from transformers import AutoConfig, AutoModelForImageClassification, ViTFeatureExtractor
from safetensors.torch import load_file

# Define local paths (assuming all files are in the same directory as this script)
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.json")
preprocessor_path = os.path.join(current_dir, "preprocessor_config.json")
model_path = os.path.join(current_dir, "model.safetensors")

# AWS S3 Configuration
BUCKET_NAME = "flowerm"
MODEL_KEY = "model.safetensors"
AWS_REGION = "us-east-1"

# Function to download model from S3
def download_model_from_s3():
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    try:
        if not os.path.exists(model_path):
            print("Downloading model from S3...")
            s3_client.download_file(BUCKET_NAME, MODEL_KEY, model_path)
            print("Model downloaded successfully.")
        else:
            print("Model already exists locally.")
    except Exception as e:
        raise Exception(f"Failed to download the model from S3: {str(e)}")

# Download the model if it doesn't exist locally
download_model_from_s3()

# Ensure necessary files exist locally
if not os.path.exists(config_path):
    raise FileNotFoundError("Model configuration file (config.json) is missing.")

# Load model configuration
try:
    config = AutoConfig.from_pretrained(config_path)
    print("Model configuration loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model configuration: {str(e)}")

# Load the preprocessor
try:
    preprocessor = ViTFeatureExtractor.from_pretrained(preprocessor_path)
    print("Preprocessor configuration loaded successfully.")
except Exception as e:
    print("Failed to load preprocessor configuration, trying to use a default preprocessor.")
    try:
        preprocessor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        preprocessor.save_pretrained(preprocessor_path)
        print("Default preprocessor configuration created successfully.")
    except Exception as e:
        raise Exception(f"Failed to create default preprocessor: {str(e)}")

# Load model weights using safetensors
try:
    state_dict = load_file(model_path)
    model = AutoModelForImageClassification.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=state_dict
    )
    print("Model loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model weights: {str(e)}")

# Label mappings
id_to_label = {
    0: 'calendula', 1: 'coreopsis', 2: 'rose', 3: 'black_eyed_susan', 4: 'water_lily', 5: 'california_poppy',
    6: 'dandelion', 7: 'magnolia', 8: 'astilbe', 9: 'sunflower', 10: 'tulip', 11: 'bellflower',
    12: 'iris', 13: 'common_daisy', 14: 'daffodil', 15: 'carnation'
}

# Function to make predictions
def predict_flower(img_path):
    """
    Predict the type of flower from the image.

    Parameters:
        img_path (str): Path to the input image.

    Returns:
        tuple: Predicted label and confidence percentage.
    """
    from PIL import Image
    import numpy as np

    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    inputs = preprocessor(images=image, return_tensors="pt")

    # Predict using the loaded model
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = id_to_label.get(predicted_class, "Unknown")

    # Only return the result if confidence is high enough
    if confidence >= 80:
        return predicted_label, confidence
    else:
        return None, None
