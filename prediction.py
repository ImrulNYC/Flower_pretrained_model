import os
import torch
import boto3
import streamlit as st
from transformers import AutoConfig, AutoModelForImageClassification, ViTFeatureExtractor
from safetensors.torch import load_file

# Define local paths
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.json")
preprocessor_path = os.path.join(current_dir, "preprocessor_config.json")
model_path = os.path.join(current_dir, "model.safetensors")

# AWS S3 Configuration using Streamlit secrets
aws_access_key_id = st.secrets["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws_secret_access_key"]
region_name = st.secrets["region_name"]

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

BUCKET_NAME = "flowerm"
MODEL_KEY = "model.safetensors"

# Function to download model from S3
def download_model_from_s3():
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

# Load model configuration
config = AutoConfig.from_pretrained(config_path)
preprocessor = ViTFeatureExtractor.from_pretrained(preprocessor_path)

# Load model weights using safetensors
state_dict = load_file(model_path)
model = AutoModelForImageClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    config=config,
    state_dict=state_dict
)

# Label mappings
id_to_label = {
    0: 'calendula', 1: 'coreopsis', 2: 'rose', 3: 'black_eyed_susan', 4: 'water_lily', 5: 'california_poppy',
    6: 'dandelion', 7: 'magnolia', 8: 'astilbe', 9: 'sunflower', 10: 'tulip', 11: 'bellflower',
    12: 'iris', 13: 'common_daisy', 14: 'daffodil', 15: 'carnation'
}

# Function to make predictions
def predict_flower(img_path):
    from PIL import Image
    import numpy as np

    image = Image.open(img_path).convert("RGB")
    inputs = preprocessor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = id_to_label.get(predicted_class, "Unknown")

    if confidence >= 80:
        return predicted_label, confidence
    else:
        return None, None


