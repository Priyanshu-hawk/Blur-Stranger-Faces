from deepface import DeepFace
import cv2
import time

img1 = cv2.imread("../test2.jpeg")
img2 = cv2.imread("../og_img.jpeg")
# rep_obj = DeepFace.extract_faces(img, detector_backend = 'retinaface')

# img_ext = DeepFace.verification.__extract_faces_and_embeddings(img, detector_backend = 'retinaface', model_name='ArcFace')

{'verified': False, 'distance': 1.015940539349389, 'threshold': 0.68, 'model': 'ArcFace', 'detector_backend': 'retinaface', 'similarity_metric': 'cosine', 'facial_areas': {'img1': {'x': 404, 'y': 159, 'w': 124, 'h': 179, 'left_eye': (495, 228), 'right_eye': (437, 236)}, 'img2': {'x': 275, 'y': 391, 'w': 144, 'h': 186, 'left_eye': (377, 464), 'right_eye': (312, 468)}}, 'time': 9.32}

img_ext = DeepFace.verify(img1, img2, model_name = 'ArcFace', detector_backend = 'retinaface')


for f_id, f_fet in img_ext['facial_areas'].items():
    if f_id == 'img1':
        cv2.rectangle(img1, (f_fet['x'], f_fet['y']), (f_fet['x']+f_fet['w'], f_fet['y']+f_fet['h']), (0, 255, 0), 2)
        cv2.imshow('img', img1)
        cv2.waitKey(0)
    else:
        cv2.rectangle(img2, (f_fet['x'], f_fet['y']), (f_fet['x']+f_fet['w'], f_fet['y']+f_fet['h']), (0, 255, 0), 2)
        cv2.imshow('img', img2)
        cv2.waitKey(0)