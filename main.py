from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from io import BytesIO
import logging

app = FastAPI()
model = YOLO("best.pt")

# Setup logging
logging.basicConfig(level=logging.INFO)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image_stream = BytesIO(image_bytes)

        results = model(image_stream)

        predictions = []

        # Handle classification
        probs = results[0].probs
        if probs is not None:
            class_names = model.names
            confidences = probs.data.cpu().numpy().tolist()

            predictions = [
                {
                    "label": class_names[i],
                    "confidence": round(conf, 2)
                }
                for i, conf in enumerate(confidences) if conf > 0.01  # confidence threshold
            ]
        else:
            # Handle object detection fallback
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = round(float(box.conf[0]), 2)
                    label = model.names.get(cls_id, f"class_{cls_id}")
                    predictions.append({
                        "label": label,
                        "confidence": conf
                    })

    except Exception as e:
        logging.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return JSONResponse(content={"predictions": predictions})

