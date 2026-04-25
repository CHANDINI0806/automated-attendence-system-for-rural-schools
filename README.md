# Automated Attendance System for Rural Schools (SDG-4)

Offline, low-resource attendance system built for Final Year BTech project work.



## Key Constraints Followed

- Fully offline (no internet/cloud APIs)
- OpenCV only (`opencv-contrib-python` for LBPH)
- Haar Cascade for face detection
- LBPH for face recognition
- SQLite local storage
- Works on Python 3.10 (Windows/macOS)

## System Requirements

- **Python 3.10** (required) – Download from [python.org](https://www.python.org/downloads/)
- Windows 10+ or macOS 10.14+
- Webcam/Camera
- ~4GB RAM minimum

## Project Structure

```
app.py
database/
	db.py
	attendance.db (auto-created)
dataset/
	known_faces/ (auto-created)
gui/
	main_window.py
models/
	lbph_model.yml (auto-created after training)
	labels.json (auto-created after training)
recognition/
	recognize.py
training/
	train_lbph.py
utils/
	config.py
	export_csv.py
	logger.py
exports/ (auto-created)
```

## SQLite Schema

Created automatically by `database/db.py`:

- `students(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, roll TEXT UNIQUE, created_at TEXT)`
- `student_images(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, image_path TEXT, captured_at TEXT, FOREIGN KEY(student_id) REFERENCES students(id))`
- `attendance(id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, date TEXT, time TEXT, status TEXT, FOREIGN KEY(student_id) REFERENCES students(id))`

Attendance de-duplication is enforced with unique index on `(student_id, date)`.

## Features

1. **Automated Student Registration + Dataset Creation**
	 - GUI inputs: Name, Roll Number, Number of photos
	 - Captures face images from webcam with delay
	 - Saves grayscale resized ROI (`200x200`) into `dataset/known_faces/<student_id>_<roll>/`
	 - Stores image paths in `student_images`

2. **LBPH Train/Update**
	 - Reads all image paths from DB
	 - Trains LBPH model
	 - Saves model to `models/lbph_model.yml`
	 - Saves label map to `models/labels.json`

3. **Live Multi-Face Attendance**
	 - Detects multiple faces per frame using Haar Cascade
	 - Recognizes via LBPH with confidence threshold
	 - Shows `Name + Roll + confidence` on frame
	 - Marks one attendance per student per day as `Present`

4. **Attendance View + CSV Export**
	 - GUI table view with optional date filtering
	 - Export report to `exports/attendance_YYYYMMDD_HHMMSS.csv`

## Run

1. Install dependencies:

```bash
pip install opencv-contrib-python numpy
```

2. Start app (use `py -3.10` for best results on Windows):

```bash
# Recommended (Windows - uses Python 3.10 specifically)
py -3.10 app.py

# Or on any platform
python app.py
```

## Reset for New Semester

To use the system for a new semester with fresh students, delete the database and model files:

```bash
# On Windows
del database\attendance.db
del models\lbph_model.yml
del models\labels.json
rmdir /s /q dataset\known_faces
rmdir /s /q exports

# On macOS/Linux
rm database/attendance.db
rm models/lbph_model.yml
rm models/labels.json
rm -rf dataset/known_faces
rm -rf exports
```

Then restart the app—it will create fresh folders and database.

## View Database Contents

To inspect what's stored in the database, use SQLite command-line or Python:

**Option 1: SQLite Browser (GUI)**
- Download [DB Browser for SQLite](https://sqlitebrowser.org/)
- Open `database/attendance.db`
- View tables: `students`, `student_images`, `attendance`

**Option 2: Python Script**
Create `view_db.py` in the project root:

```python
from database.db import get_connection

with get_connection() as conn:
    c = conn.cursor()
    
    print("=== STUDENTS ===")
    c.execute("SELECT id, name, roll, created_at FROM students")
    for row in c.fetchall():
        print(f"  ID {row[0]}: {row[1]} (Roll: {row[2]}) - Created: {row[3]}")
    
    print("\n=== STUDENT IMAGES ===")
    c.execute("SELECT student_id, COUNT(*) FROM student_images GROUP BY student_id")
    for row in c.fetchall():
        print(f"  Student {row[0]}: {row[1]} images")
    
    print("\n=== ATTENDANCE RECORDS ===")
    c.execute("SELECT COUNT(*) FROM attendance")
    total = c.fetchone()[0]
    print(f"  Total attendance records: {total}")
```

Then run:
```bash
python view_db.py
```

## Why this is better than the prototype

- Removes manual dataset folder handling
- Adds robust local database storage for students, images, and attendance
- Supports scalable re-training as more data is collected
- Provides GUI-based end-to-end workflow for demo/viva
- Adds exportable reports for school-level use

## Current Limitations

- Recognition quality drops at long distance/poor light
- No liveness detection (photo spoofing not prevented)
- Performance depends on camera quality and pose variation

## Future Work

- Add better illumination normalization
- Add anti-spoof/liveness checks
- Upgrade to deep learning-based recognition in future versions (optional, not for current offline low-resource build)
