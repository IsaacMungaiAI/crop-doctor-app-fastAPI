import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms, models

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(debug=True)

# Allow CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your mobile app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels
maize_class = {
    0: 'Blight',
    1: 'Common_Rust',
    2: 'Gray_Leaf_Spot',
    3: 'Healthy'
}

bean_classes = {
    0: "Angular Leaf Spot",
    1: "Bean Rust",
    2: "Healthy"
}

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Dynamically load only the required model
def load_model(model_type: str):
    if model_type == "maize":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(in_features=512, out_features=4)
        model.load_state_dict(torch.load("maize_model.pth", map_location="cpu"))
        classes = maize_class
    elif model_type == "bean":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(in_features=512, out_features=3)
        model.load_state_dict(torch.load("bean_disease.pth", map_location="cpu"))
        classes = bean_classes
    else:
        raise ValueError("Invalid model type")

    model.eval()
    return model, classes


# Prediction endpoint
@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_type: str = Form(...)
):
    print("Received model_type:", model_type)

    try:
        # Read and transform image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        model, classes = load_model(model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model type")

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class_name = classes.get(predicted_class_idx, "Unknown")

    prompt = f"""
    This {model_type} plant has been diagnosed with {predicted_class_name}.
    Explain in simple terms what this disease is, how it harms the crop, and how the farmer can treat and prevent it to increase yields.
    Mention any specific pesticides or herbicides that can help. If the plant is healthy, provide a good response for the same.
    """

    try:
        gemini_response = gemini_model.generate_content(prompt)
        explanation = gemini_response.text
    except Exception as e:
        explanation = f"Failed to fetch explanation: {str(e)}"

    return {
        "predicted_class_index": predicted_class_idx,
        "predicted_class_name": predicted_class_name,
        "model_used": model_type,
        "gemini_explanation": explanation
    }
