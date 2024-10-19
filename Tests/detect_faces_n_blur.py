from retinaface import RetinaFace
import cv2
import os


video_path = "./people_walk.mp4"

BASE_FOLDER = video_path.split(".")[-2].split("/")[-1]
os.makedirs(BASE_FOLDER, exist_ok=True)

BASE_OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output")
os.makedirs(BASE_OUTPUT_FOLDER, exist_ok=True)

BASE_VIDEO_FRAMES = os.path.join(BASE_FOLDER, "video_frames")
os.makedirs(BASE_VIDEO_FRAMES, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# set the video frame rate
cap.set(cv2.CAP_PROP_FPS, 30)

i = 0

while True:
    success, img = cap.read()
    if not success:
        break
    print(i)
    i += 1
    # save the frame
    cv2.imwrite(f'{BASE_VIDEO_FRAMES}/{i}.jpg', img)

cap.release()


# process the frames and blur the all the faces

for i in range(1, i):
    img = cv2.imread(f'{BASE_VIDEO_FRAMES}/{i}.jpg')
    rep_obj = RetinaFace.detect_faces(img)
    print(len(rep_obj))
    for face_id, face_fet in rep_obj.items():
        x1, y1, x2, y2 = face_fet['facial_area']
        face = img[y1:y2, x1:x2]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        img[y1:y2, x1:x2] = face
    cv2.imwrite(f'{BASE_OUTPUT_FOLDER}/{i}.jpg', img)
    print(i)
