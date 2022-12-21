from scipy.optimize import curve_fit
from read_transfer_function import *
import numpy as np


def linear_model(x, m, q):
    return m * x + q


def get_linear_gain(filepath, ch, pt, max_dacinj):

    # Get fdt data for given ch and pt
    cal_v, out = get_fdt(read_transfer_function(filepath), ch, pt)

    max_index = np.where(cal_v == max_dacinj)
    max_index = max_index[0][0]
    popt, pcov = curve_fit(linear_model, out[0:max_index], cal_v[0:max_index])

    gain = popt[0]
    pedestal = abs(popt[1])

    return gain, pedestal
