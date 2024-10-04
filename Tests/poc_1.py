from retinaface import RetinaFace
from deepface import DeepFace
from PIL import Image
import cv2
import numpy as np
import time
import os

main_img = "../og_img.jpeg"

video_path = "../out2.mp4"

os.makedirs('output', exist_ok=True)

cap = cv2.VideoCapture(video_path)
i = 0

while True:
    success, img = cap.read()
    # rep_obj = DeepFace.(img, detector_backend = 'retinaface')
    rep_obj = RetinaFace.detect_faces(img)

    # img = cv2.imread(img)

    print(rep_obj)

    for face_id, face_fet in rep_obj.items():
        cv2.rectangle(img, (face_fet['facial_area'][0], face_fet['facial_area'][1]), (face_fet['facial_area'][2], face_fet['facial_area'][3]), (0, 255, 0), 2)
        for landmark, pos in face_fet['landmarks'].items():
            cv2.circle(img, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), 10)
        
        # face id
        cv2.putText(img, face_id, (face_fet['facial_area'][0], face_fet['facial_area'][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 2)

        # blur face

        # tmp_face = img[face_fet['facial_area'][1]:face_fet['facial_area'][3], face_fet['facial_area'][0]:face_fet['facial_area'][2]]

        # img_sim = DeepFace.verify(main_img, tmp_face, model_name = 'ArcFace', detector_backend = 'retinaface')

        # if img_sim['verified'] == False:
        #     x1, y1, x2, y2 = face_fet['facial_area']
        #     face = img[y1:y2, x1:x2]
        #     face = cv2.GaussianBlur(face, (99, 99), 30)
        #     img[y1:y2, x1:x2] = face

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # write the video to a file
    # cv2.imwrite('output/{}.jpeg'.format(i), img)
    # i += 1

