import cv2
import os
import numpy as np
from datetime import datetime, time

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

ATTENDANCE_START_TIME = time(9, 0, 0)
ATTENDANCE_END_TIME = time(23, 59, 59)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {}
faces = []
ids = []
current_id = 0

# Load training images
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    labels[current_id] = name
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        ids.append(current_id)
    current_id += 1

recognizer.train(faces, np.array(ids))

marked = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        face_img = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(face_img)

        name = "Unknown"
        if confidence < 80:
            name = labels[id_]

            now = datetime.now().time()
            if name not in marked and ATTENDANCE_START_TIME <= now <= ATTENDANCE_END_TIME:
                with open(ATTENDANCE_FILE, "a") as f:
                    f.write(f"{name},{datetime.now()}\n")
                marked.add(name)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

