from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from PIL import Image
import requests
from ultralytics import YOLO

app = FastAPI()

MODEL_URL = "https://github.com/RasberryPhi/yolo-api/raw/main/best.pt"
MODEL_PATH = "best.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")

download_model()
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Only accept JPEG and PNG files (adjust as needed)
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        # Save upload to temp file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate image file
        try:
            with Image.open(temp_filename) as img:
                img.verify()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Run inference
        results = model(temp_filename)
        result = results[0]

        predictions = []

        # Classification result
        if getattr(result, "probs", None) is not None:
            class_names = model.names
            confidences = result.probs.data.cpu().numpy().tolist()
            predictions = [
                {"label": class_names[i], "confidence": round(conf, 2)}
                for i, conf in enumerate(confidences)
            ]

        # Detection fallback
        elif getattr(result, "boxes", None) is not None and getattr(result.boxes, "data", None) is not None:
            for box in result.boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                confidence = round(float(box.conf), 2)
                predictions.append({"label": label, "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    if not predictions:
        return JSONResponse(content={"message": "No predictions found."})

    return JSONResponse(content={"predictions": predictions})
