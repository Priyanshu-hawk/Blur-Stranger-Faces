from retinaface import RetinaFace
import cv2
import os
import time
import subprocess

BASE_DATA_FOLDER = os.path.abspath("./Output")
os.makedirs(BASE_DATA_FOLDER, exist_ok=True)

video_path = "./test video.mp4"
video_path = os.path.abspath(video_path)

print(video_path)

BASE_FOLDER = os.path.join(BASE_DATA_FOLDER, video_path.split(".")[0].split("/")[-1])

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

    if i == 100:
        break

cap.release()

# process the frames and blur the all the faces

# fd_model = RetinaFace.build_model()
import time

start_time = time.time()

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

print("--- %s seconds ---" % (time.time() - start_time))
print("Model pre load")

# merge the frames to a video - ffmpeg -i %d.jpg etdve.mp4
save_file = f"{BASE_FOLDER}/face_blur.mp4"
subprocess.run(["ffmpeg", "-i", f"{BASE_OUTPUT_FOLDER}/%d.jpg", save_file])