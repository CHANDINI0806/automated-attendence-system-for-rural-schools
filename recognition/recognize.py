import json
import logging
import os
from datetime import datetime

import cv2

from database.db import mark_attendance_present
from utils.config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    FACE_SIZE,
    HAAR_CASCADE_PATH,
    LABELS_PATH,
    MODEL_PATH,
)


def _load_model_and_labels():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)

    return recognizer, labels


def run_attendance_session(status_callback=None):
    if not hasattr(cv2, "face"):
        return False, "OpenCV face module is not available. Install opencv-contrib-python."

    recognizer, labels = _load_model_and_labels()
    if recognizer is None:
        return False, "Please train model first."

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return False, "Cannot open camera. Check camera permissions and index."

    if status_callback:
        status_callback("Model loaded. Attendance started. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if len(faces) == 0 and status_callback:
                status_callback("Waiting for face...")

            for (x, y, w, h) in faces:
                roi = gray[y : y + h, x : x + w]
                roi = cv2.resize(roi, FACE_SIZE)
                label, confidence = recognizer.predict(roi)

                display_text = "Unknown"
                color = (0, 0, 255)

                if confidence <= CONFIDENCE_THRESHOLD:
                    label_data = labels.get(str(label))
                    if label_data:
                        student_id = label_data["student_id"]
                        name = label_data["name"]
                        roll = label_data["roll"]
                        display_text = f"{name} ({roll}) {confidence:.1f}"
                        color = (0, 255, 0)

                        inserted = mark_attendance_present(student_id)
                        if inserted:
                            msg = f"Attendance marked: {name} ({roll})"
                            logging.info(msg)
                            if status_callback:
                                status_callback(msg)
                else:
                    display_text = f"Unknown {confidence:.1f}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    display_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Attendance - Press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if status_callback:
        status_callback("Attendance session ended.")

    return True, "Attendance session completed."
