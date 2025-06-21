from fastapi import FastAPI, File, UploadFile, HTTPException
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
        # Save uploaded file to temp
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(temp_filename)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            # No detections
            predictions = []
        else:
            predictions = []
            for box in boxes:
                # box.xyxy: bounding box coordinates (tensor)
                # box.conf: confidence score (tensor)
                # box.cls: class id (tensor)
                xyxy = box.xyxy.cpu().numpy().tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf.cpu().numpy())
                cls_id = int(box.cls.cpu().numpy())
                cls_name = model.names[cls_id] if model.names else str(cls_id)

                predictions.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return JSONResponse(content={"predictions": predictions})

