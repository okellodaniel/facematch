import logging
from helpers.file_helpers import FileHandler
from typing import Annotated,IO
from facematch import FaceMatch
from fastapi.responses import JSONResponse
from fastapi import FastAPI,HTTPException,File,status
from scalar_fastapi import get_scalar_api_reference

logging.basicConfig(level=logging.INFO)
app = FastAPI()

@app.get("/", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )
@app.post("/face-match")
async def face_match(
        img1:Annotated[bytes, File()],
        img2:Annotated[bytes, File()]
):
    try:
        if img1 is None or img2 is None:
            raise HTTPException(status_code=400, detail="img1 and img2 are required")

        filehandler = FileHandler()
        bytestoimage = filehandler.convert_bytes_to_image
        validate_file_size_type = filehandler.validate_file_size_type

        # convert bytes to Image for validation
        image_1 = bytestoimage(img1)
        image_2 = bytestoimage(img2)

        validate_file_size_type(image_1)
        validate_file_size_type(image_2)

        facematch = FaceMatch(img1, img2)
        response = facematch.match(img1, img2)

        return JSONResponse(
            content=response,
            media_type="application/json",
            status_code=200
        )
    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Error: {str(e)}'
        )
