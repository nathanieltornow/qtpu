import sys
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_cost_bench(file_path: str) -> None:
    df = pd.read_csv(file_path)
    df = df.sort_values(by="num_qubits")
    
    # filter all rows where bruteforce_cost is < 0
    df = df[df['bruteforce_cost'] >= 0]
    # df = df[df['num_qubits'] <]
    

    fig, ax = plt.subplots()
    ax.set_title("Knit Cost Benchmarks")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Cost")

    for name, group in df.groupby("name"):
        ax.plot(group["num_qubits"], group["contract_cost"], ".--", label=name)
        ax.plot(group["num_qubits"], group["bruteforce_cost"], ".-", label=name + " bruteforce")

    # ax.set_yscale("log")

    ax.legend()
    fig.savefig("knit_cost.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    match args.type:
        case "knit_cost":
            plot_cost_bench(args.file_path)
