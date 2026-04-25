import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from utils.config import DATABASE_PATH


def _ensure_db_parent_dir() -> None:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


@contextmanager
def get_connection():
    _ensure_db_parent_dir()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_unique_student_date
            ON attendance(student_id, date)
            """
        )


def add_student(name: str, roll: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students(name, roll, created_at) VALUES (?, ?, ?)",
            (name.strip(), roll.strip(), now),
        )
        return cursor.lastrowid


def get_student_by_roll(roll: str):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE roll = ?", (roll.strip(),))
        return cursor.fetchone()


def get_student_by_id(student_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        return cursor.fetchone()


def add_student_image(student_id: int, image_path: str) -> int:
    captured_at = datetime.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO student_images(student_id, image_path, captured_at) VALUES (?, ?, ?)",
            (student_id, image_path, captured_at),
        )
        return cursor.lastrowid


def get_student_images(student_id: int):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM student_images WHERE student_id = ? ORDER BY id ASC",
            (student_id,),
        )
        return cursor.fetchall()


def get_all_student_images_with_student():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                si.id AS image_id,
                si.student_id,
                si.image_path,
                s.name,
                s.roll
            FROM student_images si
            JOIN students s ON s.id = si.student_id
            ORDER BY si.id ASC
            """
        )
        return cursor.fetchall()


def mark_attendance_present(student_id: int) -> bool:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, date_str),
        )
        if cursor.fetchone():
            return False

        cursor.execute(
            "INSERT INTO attendance(student_id, date, time, status) VALUES (?, ?, ?, ?)",
            (student_id, date_str, time_str, "Present"),
        )
        return True


def get_attendance_records(start_date: str | None = None, end_date: str | None = None):
    query = """
        SELECT
            a.id,
            s.roll,
            s.name,
            a.date,
            a.time,
            a.status
        FROM attendance a
        JOIN students s ON s.id = a.student_id
    """

    params: list[str] = []
    conditions: list[str] = []

    if start_date:
        conditions.append("a.date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.date <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY a.date DESC, a.time DESC"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
