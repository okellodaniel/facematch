import filetype
import PIL.Image as Image
from io import BytesIO
from fastapi import HTTPException,status
import logging

logging.basicConfig(level=logging.INFO)

class FileHandler:
    def __init__(self) -> None:
        self.FILE_SIZE = 2097152

    def convert_bytes_to_image(self,image_bytes):
        image = Image.open(BytesIO(image_bytes))
        logging.info(f'converted bytes to image: {image.filename}')
        return image

    def validate_file_size_type(self,file):
        accepted_types = ["image/png", "image/jpeg", "image/jpg", "png", "jpeg", "jpg"]

        file_info = filetype.guess(file)
        if file_info is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unable to determine the file type"
            )
        detected_content_type = file_info.extension.lower()

        if (
                file.content_type not in accepted_types or detected_content_type not in accepted_types
        ): raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type"
        )

        real_file_size = 0
        for chunk in file.file:
            real_file_size += chunk.size
            if real_file_size > self.FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large, accepted size is less than 15mb"
                )

