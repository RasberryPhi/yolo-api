from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import uuid
import os

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    predictions = []

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(temp_filename)
        result = results[0]

        # Classification output
        if getattr(result, "probs", None) is not None:
            class_names = model.names
            confidences = result.probs.data.cpu().numpy().tolist()
            predictions = [
                {"label": class_names[i], "confidence": round(conf, 2)}
                for i, conf in enumerate(confidences)
            ]

        # Object detection output (fallback)
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


