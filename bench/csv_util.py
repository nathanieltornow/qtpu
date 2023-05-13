import csv
import os


def append_to_csv_file(filepath: str, data: dict[str, int | float]) -> None:
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as csv_file:
            csv.DictWriter(csv_file, fieldnames=data.keys()).writeheader()
            csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
        return

    with open(filepath, "a") as csv_file:
        csv.DictWriter(csv_file, fieldnames=data.keys()).writerow(data)
