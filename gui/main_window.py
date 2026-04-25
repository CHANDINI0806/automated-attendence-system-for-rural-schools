import logging
import os
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2

from database.db import (
    add_student,
    add_student_image,
    get_attendance_records,
    get_student_by_roll,
    get_student_images,
    init_db,
)
from recognition.recognize import run_attendance_session
from training.train_lbph import train_lbph_model
from utils.config import (
    CAMERA_INDEX,
    CAPTURE_DELAY_SECONDS,
    FACE_SIZE,
    HAAR_CASCADE_PATH,
    KNOWN_FACES_DIR,
    ensure_project_dirs,
)
from utils.export_csv import export_attendance_to_csv
from utils.logger import setup_logger


def _sanitize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text.strip())


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Automated Attendance System (Offline)")
        self.root.geometry("750x460")

        self.status_var = tk.StringVar(value="System ready")

        self.name_var = tk.StringVar()
        self.roll_var = tk.StringVar()
        self.photos_var = tk.StringVar(value="10")

        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self.root,
            text="Automated Attendance System for Rural Schools",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(pady=10)

        form = ttk.Frame(self.root)
        form.pack(fill="x", padx=20, pady=10)

        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(form, textvariable=self.name_var, width=30).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(form, text="Roll Number").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(form, textvariable=self.roll_var, width=30).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(form, text="No. of Photos").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(form, textvariable=self.photos_var, width=30).grid(row=2, column=1, padx=5, pady=5)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(btn_frame, text="Capture Photos", command=self.capture_photos).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(btn_frame, text="Train/Update Model", command=self.train_model).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(btn_frame, text="Start Attendance", command=self.start_attendance).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(btn_frame, text="View Attendance", command=self.view_attendance).grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(btn_frame, text="Export Attendance CSV", command=self.export_attendance).grid(row=0, column=4, padx=6, pady=6)

        status_box = ttk.LabelFrame(self.root, text="Status")
        status_box.pack(fill="both", expand=True, padx=20, pady=10)

        self.status_label = ttk.Label(status_box, textvariable=self.status_var, wraplength=700, justify="left")
        self.status_label.pack(anchor="w", padx=10, pady=10)

    def set_status(self, message: str):
        self.status_var.set(message)
        logging.info(message)
        self.root.update_idletasks()

    def _validate_registration(self):
        name = self.name_var.get().strip()
        roll = self.roll_var.get().strip()
        photo_count_raw = self.photos_var.get().strip()

        if not name:
            messagebox.showerror("Validation Error", "Name is required.")
            return None
        if not roll:
            messagebox.showerror("Validation Error", "Roll Number is required.")
            return None

        try:
            photo_count = int(photo_count_raw)
            if photo_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Validation Error", "Number of photos must be a positive integer.")
            return None

        return name, roll, photo_count

    def capture_photos(self):
        validated = self._validate_registration()
        if validated is None:
            return

        name, roll, photo_count = validated

        existing = get_student_by_roll(roll)
        if existing:
            add_more = messagebox.askyesno(
                "Student already registered",
                "Student already registered. Do you want to add more photos?",
            )
            if not add_more:
                return
            student_id = existing["id"]
            student_name = existing["name"]
        else:
            try:
                student_id = add_student(name, roll)
                student_name = name
            except Exception as exc:
                messagebox.showerror("Database Error", f"Failed to register student: {exc}")
                return

        student_dir = os.path.join(KNOWN_FACES_DIR, f"{student_id}_{_sanitize_text(roll)}")
        os.makedirs(student_dir, exist_ok=True)

        existing_images = get_student_images(student_id)
        next_index = len(existing_images) + 1

        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to open camera.")
            return

        prompts = ["Look straight", "Turn slightly left", "Turn slightly right"]
        prompt_index = 0

        captured = 0
        last_capture_time = 0.0
        self.set_status("Camera opened. Waiting for face...")

        try:
            while captured < photo_count:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                faces_sorted = sorted(faces, key=lambda item: item[2] * item[3], reverse=True)
                if faces_sorted:
                    x, y, w, h = faces_sorted[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    now = time.time()

                    if now - last_capture_time >= CAPTURE_DELAY_SECONDS:
                        roi = gray[y : y + h, x : x + w]
                        roi = cv2.resize(roi, FACE_SIZE)

                        filename = f"img_{next_index:03d}.jpg"
                        image_path = os.path.join(student_dir, filename)
                        cv2.imwrite(image_path, roi)
                        add_student_image(student_id, image_path)

                        captured += 1
                        next_index += 1
                        last_capture_time = now

                        self.set_status(f"Captured {captured}/{photo_count} for {student_name} ({roll}). Next photo in 40 seconds...")
                        prompt_index = (prompt_index + 1) % len(prompts)
                    else:
                        seconds_left = int(CAPTURE_DELAY_SECONDS - (now - last_capture_time))
                        self.set_status(f"Next photo in {seconds_left}s | Captured {captured}/{photo_count} for {student_name} ({roll})")
                else:
                    self.set_status("Waiting for face...")

                now = time.time()
                seconds_left = int(CAPTURE_DELAY_SECONDS - (now - last_capture_time)) if captured > 0 else 0
                if seconds_left < 0:
                    seconds_left = 0

                cv2.putText(
                    frame,
                    f"{prompts[prompt_index]} | {captured}/{photo_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Next photo in: {seconds_left}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press q to cancel",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Capture Student Photos", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if captured == 0:
            self.set_status("No photos captured.")
            return

        self.set_status(f"Capture complete: {captured} photo(s) saved for {student_name} ({roll})")
        messagebox.showinfo("Capture Complete", f"Saved {captured} photo(s) successfully.")

    def train_model(self):
        self.set_status("Training model...")
        success, message = train_lbph_model()
        self.set_status(message)
        if success:
            messagebox.showinfo("Training", message)
        else:
            messagebox.showwarning("Training", message)

    def start_attendance(self):
        def _run():
            success, message = run_attendance_session(status_callback=self.set_status)
            if success:
                messagebox.showinfo("Attendance", message)
            else:
                messagebox.showwarning("Attendance", message)

        self.set_status("Starting attendance...")
        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

    def view_attendance(self):
        window = tk.Toplevel(self.root)
        window.title("Attendance Records")
        window.geometry("760x450")

        filter_frame = ttk.Frame(window)
        filter_frame.pack(fill="x", padx=10, pady=10)

        start_var = tk.StringVar()
        end_var = tk.StringVar()

        ttk.Label(filter_frame, text="Start Date (YYYY-MM-DD)").grid(row=0, column=0, padx=4, pady=4)
        ttk.Entry(filter_frame, textvariable=start_var, width=16).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(filter_frame, text="End Date (YYYY-MM-DD)").grid(row=0, column=2, padx=4, pady=4)
        ttk.Entry(filter_frame, textvariable=end_var, width=16).grid(row=0, column=3, padx=4, pady=4)

        columns = ("roll", "name", "date", "time", "status")
        tree = ttk.Treeview(window, columns=columns, show="headings", height=15)
        tree.heading("roll", text="Roll")
        tree.heading("name", text="Name")
        tree.heading("date", text="Date")
        tree.heading("time", text="Time")
        tree.heading("status", text="Status")

        tree.column("roll", width=100)
        tree.column("name", width=180)
        tree.column("date", width=120)
        tree.column("time", width=120)
        tree.column("status", width=100)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def load_records():
            for item in tree.get_children():
                tree.delete(item)

            rows = get_attendance_records(
                start_date=start_var.get().strip() or None,
                end_date=end_var.get().strip() or None,
            )
            for row in rows:
                tree.insert("", "end", values=(row["roll"], row["name"], row["date"], row["time"], row["status"]))

        ttk.Button(filter_frame, text="Load", command=load_records).grid(row=0, column=4, padx=6)
        load_records()

    def export_attendance(self):
        start_date = simpledialog.askstring(
            "Export Attendance",
            "Start Date (YYYY-MM-DD) or leave blank for all:",
            parent=self.root,
        )
        if start_date is None:
            return

        end_date = simpledialog.askstring(
            "Export Attendance",
            "End Date (YYYY-MM-DD) or leave blank for all:",
            parent=self.root,
        )
        if end_date is None:
            return

        path, count = export_attendance_to_csv(
            start_date=start_date.strip() or None,
            end_date=end_date.strip() or None,
        )

        if count == 0:
            messagebox.showinfo("Export", "No attendance records found for selected filters.")
            self.set_status("Export skipped: no records found.")
            return

        messagebox.showinfo("Export", f"Exported {count} records to:\n{path}")
        self.set_status(f"Export complete: {count} records saved.")


def launch_app():
    setup_logger()
    ensure_project_dirs()
    init_db()

    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
