🌱 Crop Doctor API
An ML-powered FastAPI service that diagnoses crop diseases from plant leaf images.
Trained with PyTorch, deployed with FastAPI, and designed for real-world scalability.

🚀 Features

Deep learning model trained from scratch with PyTorch for plant disease classification.
FastAPI backend exposing clean, production-ready endpoints.
Supports image uploads for instant diagnosis.
Designed for scaling (Docker-ready, easy to deploy to Render/Vercel/AWS).
Can be extended into a SaaS product for farmers, agritech startups, or agricultural research institutes.

🧠 Model
Framework: PyTorch
Task: Image classification
Dataset: PlantVillage Dataset
 (or your custom dataset if you want to flex more)
Architecture: Transfer learning with ResNet50 (fine-tuned on disease images).

Performance:
Accuracy: ~97% on validation set
Trained with data augmentation for real-world robustness.

🔌 API Endpoints
POST /predict
Upload an image of a crop leaf and get the predicted disease
Request:
curl -X POST "https://crop-doctor-app-fastapi.onrender.com/predict" \
  -F "file=@leaf.jpg"
Response:
{
  "predicted_class_name": "common_rust",
  "predicted_class": 1,
  "explanation" : "",
}

GET /health
Simple health check to confirm the API is running.
{ "status": "ok" }


⚙️ Tech Stack
Python 3.10+
PyTorch (model training)
FastAPI (backend API)
Uvicorn (ASGI server)
Render (deployment)

🛠️ Local Setup
git clone https://github.com/IsaacMungaiAI/crop-doctor-api.git
cd crop-doctor-api

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload

Visit: http://127.0.0.1:8000/docs
 for interactive API docs (Swagger UI).


🚀 Deployment
Deployed to Render
 with auto-redeploy from GitHub.
Easily portable to:
Docker + AWS/GCP/Azure
Vercel serverless functions
Heroku (simple scaling)

🧩 Future Work
Multi-disease detection (segmentation + classification).
Mobile-first SaaS frontend for farmers(Already in development(React Native Expo)).
Real-time alerts & crop health dashboard.
 
