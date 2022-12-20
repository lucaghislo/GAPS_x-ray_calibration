import os.path
import pandas as pd
import numpy as np


def read_pedestals(filepath):
    pedestal_data = pd.read_csv(filepath, sep="\t")

    return pedestal_data


def get_pedestal(pedestal_data, ch, pt):
    pedestal_subset = pedestal_data[
        (pedestal_data["ch"] == ch) & (pedestal_data["pt"] == pt)
    ]

    return float(pedestal_subset["mean"])
