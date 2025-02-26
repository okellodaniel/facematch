from fastapi import FastAPI,HTTPException,File
from facematch import FaceMatch
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
app = FastAPI()

facematch = FaceMatch()

image_extensions = ["jpg", "jpeg", "png"]

@app.post("/face-match")
async def facematch(img1: File(), img2: File()):
    try:
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="img1 and img2 are required")
        response = facematch.faceMatch(img1, img2)
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=400, detail=str(e))
    return response
