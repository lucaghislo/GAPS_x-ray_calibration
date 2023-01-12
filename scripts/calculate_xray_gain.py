from scipy.optimize import curve_fit
from read_transfer_function import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

conv_factor = 0.841


def linear_model(x, m, q):
    return m * x + q


def cubic_model(x, q, m1, m2, m3):
    return q + m1 * x + m2 * (x ** 2) + m3 * (x ** 3)


# Linear interpolation
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

    if outpath != "":
        # Plot of interpolation goodness
        plt.clf()
        out_selected_show = out[1 : max_index + 7]
        cal_v_kev_selected_show = cal_v_kev[1 : max_index + 7]
        plt.plot(
            out_selected_show, cal_v_kev_selected_show, marker="o", linestyle="None"
        )
        plt.plot(out_selected, linear_model(out_selected, *popt))
        plt.xlabel("Channel Output [ADU]")
        plt.ylabel("Incoming Energy [keV]")
        plt.title("\\textbf{X-ray region linear interpolation}")

        matplotlib.pyplot.text(
            200,
            max(cal_v_kev_selected_show),
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

        plt.savefig(outpath)

    return gain, pedestal


# Cubic interpolation
def get_cubic_gain(filepath, ch, pt, max_dacinj, outpath=""):

    # Get fdt data for given ch and pt
    cal_v, out = get_fdt(read_transfer_function(filepath), ch, pt)

    cal_v_kev = [cal_v_i * conv_factor for cal_v_i in cal_v]

    max_index = np.where(cal_v == max_dacinj)
    max_index = max_index[0][0]
    out_selected = out[1:max_index]
    cal_v_kev_selected = cal_v_kev[1:max_index]
    popt, pcov = curve_fit(cubic_model, out_selected, cal_v_kev_selected)

    gain = popt[1]
    pedestal = abs(popt[0])

    if outpath != "":
        # Plot of interpolation goodness
        plt.clf()
        out_selected_show = out[1 : max_index + 7]
        cal_v_kev_selected_show = cal_v_kev[1 : max_index + 7]
        plt.plot(
            out_selected_show, cal_v_kev_selected_show, marker="o", linestyle="None"
        )
        plt.plot(out_selected, cubic_model(out_selected, *popt))
        plt.xlabel("Channel Output [ADU]")
        plt.ylabel("Incoming Energy [keV]")
        plt.title("\\textbf{X-ray region cubic interpolation}")

        matplotlib.pyplot.text(
            200,
            max(cal_v_kev_selected_show),
            "Estimated cubic model\n"
            + str(cal_v_kev_selected[0])
            + " keV - "
            + str(cal_v_kev_selected[len(cal_v_kev_selected) - 1])
            + " keV\n$y="
            + str(np.round(popt[0], 2))
            + " + "
            + str(np.round(popt[1], 3))
            + " \cdot x "
            + str(np.round(popt[2], 3))
            + " \cdot x^{2} + "
            + str(np.round(popt[3], 6))
            + " \cdot x^{3}$",
            fontsize=12,
            verticalalignment="top",
        )

        plt.savefig(outpath)

    return gain, pedestal


# Linear interpolation with ADU as output
def get_linear_gain_realfdt(filepath, ch, pt, max_dacinj, outpath=""):

    # Get fdt data for given ch and pt
    cal_v, out = get_fdt(read_transfer_function(filepath), ch, pt)

    cal_v_kev = [cal_v_i * conv_factor for cal_v_i in cal_v]

    max_index = np.where(cal_v == max_dacinj)
    max_index = max_index[0][0]
    out_selected = out[1:max_index]
    cal_v_kev_selected = cal_v_kev[1:max_index]
    popt, pcov = curve_fit(linear_model, cal_v_kev_selected, out_selected)

    gain = popt[0]
    pedestal = abs(popt[1])

    if outpath != "":
        # Plot of interpolation goodness
        plt.clf()
        out_selected_show = out[1 : max_index + 7]
        cal_v_kev_selected_show = cal_v_kev[1 : max_index + 7]
        plt.plot(
            cal_v_kev_selected_show, out_selected_show, marker="o", linestyle="None"
        )

        cal_v_kev_selected = np.array(cal_v_kev_selected)

        plt.plot(cal_v_kev_selected, linear_model(cal_v_kev_selected, *popt))
        plt.ylabel("Channel Output [ADU]")
        plt.xlabel("Incoming Energy [keV]")
        plt.title("\\textbf{X-ray region linear interpolation}")

        matplotlib.pyplot.text(
            200,
            max(cal_v_kev_selected_show),
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

        plt.savefig(outpath)

    return gain, pedestal


# Cubic interpolation with ADU as output
def get_cubic_gain_realfdt(filepath, ch, pt, max_dacinj, outpath=""):

    # Get fdt data for given ch and pt
    cal_v, out = get_fdt(read_transfer_function(filepath), ch, pt)

    cal_v_kev = [cal_v_i * conv_factor for cal_v_i in cal_v]

    max_index = np.where(cal_v == max_dacinj)
    max_index = max_index[0][0]
    out_selected = out[1:max_index]
    cal_v_kev_selected = cal_v_kev[1:max_index]
    popt, pcov = curve_fit(cubic_model, cal_v_kev_selected, out_selected)

    gain = popt[1]
    pedestal = abs(popt[0])

    if outpath != "":
        # Plot of interpolation goodness
        plt.clf()
        out_selected_show = out[1 : max_index + 7]
        cal_v_kev_selected_show = cal_v_kev[1 : max_index + 7]
        plt.plot(
            cal_v_kev_selected_show, out_selected_show, marker="o", linestyle="None"
        )
        cal_v_kev_selected = np.array(cal_v_kev_selected)
        plt.plot(cal_v_kev_selected, cubic_model(cal_v_kev_selected, *popt))
        plt.ylabel("Channel Output [ADU]")
        plt.xlabel("Incoming Energy [keV]")
        plt.title("\\textbf{X-ray region cubic interpolation}")

        matplotlib.pyplot.text(
            200,
            max(cal_v_kev_selected_show),
            "Estimated cubic model\n"
            + str(cal_v_kev_selected[0])
            + " keV - "
            + str(cal_v_kev_selected[len(cal_v_kev_selected) - 1])
            + " keV\n$y="
            + str(np.round(popt[0], 2))
            + " + "
            + str(np.round(popt[1], 3))
            + " \cdot x "
            + str(np.round(popt[2], 3))
            + " \cdot x^{2} + "
            + str(np.round(popt[3], 6))
            + " \cdot x^{3}$",
            fontsize=12,
            verticalalignment="top",
        )

        plt.savefig(outpath)

    return gain, pedestal
