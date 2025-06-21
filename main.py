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

        boxes = results[0].boxes
        if boxes is not None:
            predictions = boxes.data.cpu().numpy().tolist()
        else:
            predictions = []

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return JSONResponse(content={"predictions": predictions})
