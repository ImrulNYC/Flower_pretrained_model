import os
import requests
import torch
from transformers import AutoConfig, AutoModelForImageClassification, ViTImageProcessor
from safetensors.torch import load_file

# Updated URLs to GitHub raw files
CONFIG_URL = "https://raw.githubusercontent.com/ImrulNYC/Flower_pretrained_model/15e9f3e9b7419b1dbc4aeea2bd7d3751322270cc/config.json"
PREPROCESSOR_URL = "https://raw.githubusercontent.com/ImrulNYC/Flower_pretrained_model/15e9f3e9b7419b1dbc4aeea2bd7d3751322270cc/preprocessor_config.json"

# Local paths to save files
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.safetensors")  # This still comes from S3
config_path = os.path.join(current_dir, "config.json")
preprocessor_path = os.path.join(current_dir, "preprocessor_config.json")

# Function to download file from a URL
def download_file_from_url(url, local_path):
    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(local_path)} successfully.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download {os.path.basename(local_path)} from URL: {str(e)}")

# Download config.json and preprocessor_config.json from GitHub
download_file_from_url(CONFIG_URL, config_path)
download_file_from_url(PREPROCESSOR_URL, preprocessor_path)

# AWS S3 Configuration (model file only)
BUCKET_NAME = "flowerm"
MODEL_KEY = "model.safetensors"
AWS_REGION = "us-east-1"

# Function to download model from S3
def download_file_from_s3(bucket_name, s3_key, local_path):
    import boto3
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    try:
        print(f"Downloading {s3_key} from S3...")
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"{s3_key} downloaded successfully.")
    except Exception as e:
        raise Exception(f"Failed to download {s3_key} from S3: {str(e)}")

# Download the model file from S3
download_file_from_s3(BUCKET_NAME, MODEL_KEY, model_path)

# Load model configuration
try:
    config = AutoConfig.from_pretrained(config_path)
    print("Model configuration loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model configuration: {str(e)}")

# Load the preprocessor
try:
    preprocessor = ViTImageProcessor.from_pretrained(preprocessor_path)
    print("Preprocessor configuration loaded successfully.")
except Exception as e:
    print("Failed to load preprocessor configuration, trying to use a default preprocessor.")
    try:
        preprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
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
