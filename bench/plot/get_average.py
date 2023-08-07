
import pandas as pd
import numpy as np

def get_average(dataframes: list[pd.DataFrame], key: str, base_key: str | None = None) -> float:
    if base_key is not None:
        for df in dataframes:
            df["for_mean"] = df[key] / df[base_key]

    # concatenate all dataframes into one
    df = pd.concat(dataframes, ignore_index=True)
    print("median:", np.median(df["for_mean"]))
    print("mean:", np.average(df["for_mean"]))
    print("std:", np.std(df["for_mean"]))
    print("min:", np.min(df["for_mean"]))
    print("max:", np.max(df["for_mean"]))
    print()
    
    