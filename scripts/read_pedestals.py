import os.path
import pandas as pd
import numpy as np


def read_pedestals(filepath):
    pedestal_data = pd.read_csv(filepath)
    print(pedestal_data)
