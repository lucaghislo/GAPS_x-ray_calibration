import os.path
import pandas as pd
import numpy as np


def read_transfer_function(filepath):
    fdt_data = pd.read_csv(filepath, sep="\t")

    return fdt_data


def get_fdt(fdt_data, ch, pt):
    fdt_subset = fdt_data[fdt_data["ch"] == ch]
    fdt_subset = fdt_subset[fdt_subset["pt"] == pt]

    cal_v = fdt_subset["CAL_V"]
    out = fdt_subset["mean"]

    cal_v = cal_v.to_numpy()
    out = out.to_numpy()

    return cal_v, out
