import json
import logging
import os

import cv2
import numpy as np

from database.db import get_all_student_images_with_student
from utils.config import FACE_SIZE, LABELS_PATH, MODEL_PATH, ensure_project_dirs


def train_lbph_model() -> tuple[bool, str]:
    ensure_project_dirs()

    if not hasattr(cv2, "face"):
        return False, "OpenCV face module is not available. Install opencv-contrib-python."

    image_rows = get_all_student_images_with_student()
    if not image_rows:
        return False, "No student images found. Please register students and capture photos first."

    student_to_label: dict[int, int] = {}
    label_details: dict[str, dict] = {}
    next_label = 0

    faces: list[np.ndarray] = []
    labels: list[int] = []

    for row in image_rows:
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            logging.warning("Skipping missing image: %s", image_path)
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.warning("Skipping unreadable image: %s", image_path)
            continue

        resized = cv2.resize(image, FACE_SIZE)
        student_id = row["student_id"]

        if student_id not in student_to_label:
            student_to_label[student_id] = next_label
            label_details[str(next_label)] = {
                "student_id": student_id,
                "name": row["name"],
                "roll": row["roll"],
            }
            next_label += 1

        faces.append(resized)
        labels.append(student_to_label[student_id])

    if not faces:
        return False, "No valid images available for training."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as labels_file:
        json.dump(label_details, labels_file, indent=2)

    return True, f"Model trained successfully with {len(faces)} images."
