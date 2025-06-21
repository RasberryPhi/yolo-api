from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import uuid
import os

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        # Save uploaded file to temp location
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # === ADD IMAGE VALIDATION HERE ===
        try:
            with Image.open(temp_filename) as img:
                img.verify()  # Check if image is valid (raises if corrupted/unsupported)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        # Now run model inference, knowing file is valid image
        results = model(temp_filename)
        result = results[0]

        predictions = []
        # Your existing classification and detection logic here
        if getattr(result, "probs", None) is not None:
            # classification code ...
            pass
        elif getattr(result, "boxes", None) is not None and getattr(result.boxes, "data", None) is not None:
            # detection code ...
            pass

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # Return predictions or no prediction message
    if not predictions:
        return JSONResponse(content={"message": "No predictions found."})
    return JSONResponse(content={"predictions": predictions})


