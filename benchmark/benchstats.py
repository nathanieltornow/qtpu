import csv


def bench_results_from_csv(csv_path: str) -> list[dict[str, float]]:
    """
    Parses the benchmark results from a csv file.
    """
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]  # type: ignore

import os

# current filepath
dirname = os.path.dirname(__file__)

print(bench_results_from_csv("bench_results/ham_sim_03-23-17-43-45.csv"))