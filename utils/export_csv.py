import csv
from datetime import datetime
import os

from database.db import get_attendance_records
from utils.config import EXPORTS_DIR


def export_attendance_to_csv(start_date: str | None = None, end_date: str | None = None):
    records = get_attendance_records(start_date=start_date, end_date=end_date)
    if not records:
        return None, 0

    os.makedirs(EXPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(EXPORTS_DIR, f"attendance_{stamp}.csv")

    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Roll", "Name", "Date", "Time", "Status"])
        for row in records:
            writer.writerow([row["roll"], row["name"], row["date"], row["time"], row["status"]])

    return file_path, len(records)
