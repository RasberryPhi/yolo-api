from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()
model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(temp_filename)

        predictions = []
        probs = results[0].probs  # classification probabilities
        if probs is not None:
            class_names = model.names
            confidences = probs.data.cpu().numpy().tolist()
            predictions = [
                {"label": class_names[i], "confidence": round(conf, 2)}
                for i, conf in enumerate(confidences)
            ]
        else:
            # fallback in case it's object detection instead of classification
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    label = model.names[int(box.cls)]
                    confidence = round(float(box.conf), 2)
                    predictions.append({"label": label, "confidence": confidence})

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return JSONResponse(content={"predictions": predictions})

                    "class_id": cls_id,
                    "class_name": cls_name
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return JSONResponse(content={"predictions": predictions})

