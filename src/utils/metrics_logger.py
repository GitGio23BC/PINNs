import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, filepath: str | Path, fieldnames: list[str]):
        self.filepath = Path(filepath)
        self.fieldnames = fieldnames

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        #if not self.filepath.exists():
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, metrics: dict):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)