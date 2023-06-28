import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def _plot(ax, num_key: str, denom_key: str, dataframes: dict[str, pd.DataFrame]):
    key = f"rel_{num_key}/{denom_key}"
    for df in dataframes.values():
        df[key] = df[num_key] / df[denom_key]

    for name, df in dataframes.items():
        grouped_df = (
            df.groupby("num_qubits")
            .agg({key: ["mean", "sem"]})
            .sort_values(by=["num_qubits"])
            .reset_index()
        )

        x = grouped_df["num_qubits"]
        y_mean = grouped_df[key]["mean"]
        y_error = grouped_df[key]["sem"]

        ax.errorbar(
            x,
            y_mean,
            yerr=y_error,
            label=name,
            marker="o",
            linestyle="dashed",
            linewidth=2,
        )
    # plot a horizontal line at 1
    ax.axhline(y=1, color="black", linewidth=2)


if __name__ == "__main__":
    files = {
        # "vqe_1": "results/gen_bisect/fake_montreal_v2/2_vgates/hamsim_1.csv",
        # "vqe_2": "results/gen_bisect/fake_montreal_v2/2_vgates/hamsim_2.csv",
        # "vqe_3": "results/gen_bisect/fake_montreal_v2/2_vgates/hamsim_3.csv",
        # "vqe_2": "results/gen_bisect/fake_montreal_v2/1_vgates/qaoa_0.2.csv",
        # "qaoa_0.1": "results/gen_bisect/fake_montreal_v2/1_vgates/qaoa_0.1.csv",
        # "qaoa_0.2": "results/gen_bisect/fake_montreal_v2/2_vgates/qaoa_0.2.csv",
        # "qaoa_0.3": "results/gen_bisect/fake_montreal_v2/3_vgates/qaoa_0.3.csv",
        # # "qaoa_0.3": "results/gen_bisect/fake_montreal_v2/1_vgates/qaoa_0.3.csv",
        # "vqe_3": "results/gen_bisect/fake_montreal_v2/3_vgates/qaoa_0.3.csv",
        # "vqe_2": "results/vqr/fake_montreal_v2/3_distance/vqe_2.csv",
        # "vqe_3": "results/vqr/fake_montreal_v2/3_distance/vqe_3.csv",
        # "qaoa_0.1": "results/vqr/fake_montreal_v2/4_distance/qaoa_0.1.csv",
        # "qaoa_0.2": "results/vqr/fake_montreal_v2/4_distance/qaoa_0.2.csv",
        # "qaoa_0.3": "results/vqr/fake_montreal_v2/4_distance/qaoa_0.3.csv",
        # "hamsim_1": "results/gen_bisect/fake_montreal_v2/3_vgates/hamsim_1.csv",
        # "hamsim_2": "results/gen_bisect/fake_montreal_v2/3_vgates/hamsim_2.csv",
        # "hamsim_3": "results/gen_bisect/fake_montreal_v2/3_vgates/hamsim_3.csv",
        
        # "vqe_1": "results/gen_bisect/fake_montreal_v2/4_vgates/vqe_1.csv",
        # "vqe_2": "results/gen_bisect/fake_montreal_v2/4_vgates/vqe_2.csv",
        "vqe_3": "results/gen_bisect/fake_montreal_v2/4_vgates/vqe_3.csv",
        "vqe_4": "results/gen_bisect/fake_montreal_v2/4_vgates/vqe_4.csv",
        
        # "qaoa_3": "results/vqr/fake_montreal_v2/3_distance/vqe_3.csv",
        # "vqe_1": "results/vqr/fake_montreal_v2/3_distance/vqe_1.csv",
    }

    dataframes = {name: pd.read_csv(file) for name, file in files.items()}

    fig, ax = plt.subplots(figsize=(8, 6))

    # _plot(ax, "depth_base", "depth", dataframes)
    _plot(ax, "num_cnots_base", "num_cnots", dataframes)

    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Relative CNOT Count")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    plt.show()
