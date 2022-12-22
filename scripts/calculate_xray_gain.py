from scipy.optimize import curve_fit
from read_transfer_function import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

conv_factor = 0.841


def linear_model(x, m, q):
    return m * x + q


def get_linear_gain(filepath, ch, pt, max_dacinj, outpath=""):

    # Get fdt data for given ch and pt
    cal_v, out = get_fdt(read_transfer_function(filepath), ch, pt)

    cal_v_kev = [cal_v_i * conv_factor for cal_v_i in cal_v]

    max_index = np.where(cal_v == max_dacinj)
    max_index = max_index[0][0]
    out_selected = out[1:max_index]
    cal_v_kev_selected = cal_v_kev[1:max_index]
    popt, pcov = curve_fit(linear_model, out_selected, cal_v_kev_selected)

    gain = popt[0]
    pedestal = abs(popt[1])

    # Plot of interpolation goodness
    plt.clf()
    out_selected_show = out[1 : max_index + 7]
    cal_v_kev_selected_show = cal_v_kev[1 : max_index + 7]
    plt.plot(out_selected_show, cal_v_kev_selected_show, marker="o", linestyle="None")
    plt.plot(out_selected, linear_model(out_selected, *popt))
    plt.xlabel("Channel Output [ADU]")
    plt.ylabel("Incoming Energy [keV]")
    plt.title("\\textbf{X-ray region linear interpolation}")

    matplotlib.pyplot.text(
        200,
        700,
        "Estimated linear model\n"
        + str(cal_v_kev_selected[0])
        + " keV - "
        + str(cal_v_kev_selected[len(cal_v_kev_selected) - 1])
        + " keV\n$y="
        + str(np.round(gain, 3))
        + " \cdot x + "
        + str(np.round(pedestal, 2))
        + "$",
        fontsize=12,
        verticalalignment="top",
    )

    if outpath != "":
        plt.savefig(outpath)

    return gain, pedestal
