import cv2
import base64
import insightface
import logging
import numpy as np
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.DEBUG)

class FaceMatch:
    def __init__(self, selfiepath: str, idcardpath: str ):
        self.selfiepath = selfiepath
        self.idcardpath = idcardpath
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'],det_thresh=0.5)
        self.THRESHOLD=0.5
    def prepare_model(self):
        self.app.prepare(ctx_id=0,det_size=(640,640))
    def _draw_detections(self,img):
        img = cv2.imread(img)
        faces = self.app.get(img)
        if len(faces) == 0:
            logging.info("No face detected")
            return
        for face in faces:
            bbox = face.bbox.astype(np.int)

        cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,0), 2)

        _,encoded_img = cv2.imencode('.jpg', img)
        base64_img = base64.b64encode(encoded_img).decode('utf-8')

        return base64_img
    def get_face_embeddings(self,img):
        img = cv2.imread(img)
        faces = self.app.get(img)
        if len(faces) == 0:
            logging.info("No face detected")
            return
        face = sorted(
            faces,
            key=lambda face: (face.bbox.shape[2] - face.bbox.shape[0] - face.bbox.shape[1]),
            reverse=True
        )[0]

        return face
    def _cosine_similarity(self,embedding1,embedding2):
        return np.dot(embedding1,embedding2) / (np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

