# utils/csv_logger.py
import os
import csv

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        # Keep only known fields; fill missing with empty
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)
