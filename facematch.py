import cv2
import base64
import logging
import numpy as np
import json
from PIL import Image
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.DEBUG)

class FaceMatch:
    def __init__(self, selfiepath, idcardpath):
        self.selfiepath = selfiepath
        self.idcardpath = idcardpath
        self.THRESHOLD=0.5
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'],
                                det_thresh=0.5)

    def _draw_detections(self,img):
        try:
            img_array = np.asarray(bytearray(img), dtype="uint8")
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            logging.info(f'read file {img}')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            faces = self.app.get(img)
            logging.info(f'read faces {len(faces)}')

            if len(faces) == 0:
                logging.info("No face detected")
                return
            for face in faces:
                bbox = face.bbox.astype('int')

            cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (121,226,219), 2)

            _,encoded_img = cv2.imencode('.jpg', img)
            base64_img = base64.b64encode(encoded_img).decode('utf-8')

            return base64_img
        except Exception as e:
            logging.error(e)

    def _get_face_embeddings(self,img):
        img_array = np.asarray(bytearray(img), dtype="uint8")
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        logging.info(f'read file {img}')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        faces = self.app.get(img)
        if len(faces) == 0:
            logging.info("No face detected")
            return
        face = sorted(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0] - face.bbox[1]),
            reverse=True
        )[0]
        logging.info('face :',len(face))
        return face.embedding

    def _cosine_similarity(self,embedding1,embedding2):
        return np.dot(embedding1,embedding2) / (np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

    def match(self,img1,img2):
        self.selfiepath = img1
        self.idcardpath = img2

        selfie_base64 = self._draw_detections(self.selfiepath)
        idcard_base64 = self._draw_detections(self.idcardpath)

        selfie_embedding = self._get_face_embeddings(self.selfiepath)
        idcard_embedding = self._get_face_embeddings(self.idcardpath)

        logging.info(f'selfie embedding: {selfie_embedding}')
        logging.info(f'idcard embedding: {idcard_embedding}')

        similarity = self._cosine_similarity(selfie_embedding,idcard_embedding)

        # calculate similarity based on a threshold
        ismatch = similarity >= self.THRESHOLD

        return {
            "similarity": float(similarity),
            "ismatch": bool(ismatch),
            "selfie": selfie_base64,
            "idcard": idcard_base64
        }



