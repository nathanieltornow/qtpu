import csv

import numpy as np


def results_from_csv(csv_path: str) -> list[dict[str, float]]:
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]  # type: ignore


def get_results(
    results: list[dict[str, float]], bench_key: str
) -> dict[int, list[float]]:
    results_dict: dict[int, list[float]] = {}
    for result in results:
        num_qubits = int(result["num_qubits"])
        if num_qubits not in results_dict:
            results_dict[num_qubits] = []
        results_dict[num_qubits].append(float(result[bench_key]))
    return results_dict
