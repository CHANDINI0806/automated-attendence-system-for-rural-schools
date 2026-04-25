import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "attendance.db")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
KNOWN_FACES_DIR = os.path.join(DATASET_DIR, "known_faces")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lbph_model.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

CAMERA_INDEX = 0
FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 70.0
CAPTURE_DELAY_SECONDS = 5

HAAR_CASCADE_PATH = os.path.join(
    __import__("cv2").data.haarcascades,
    "haarcascade_frontalface_default.xml",
)


def ensure_project_dirs() -> None:
    for path in [DATABASE_DIR, KNOWN_FACES_DIR, MODELS_DIR, EXPORTS_DIR]:
        os.makedirs(path, exist_ok=True)
