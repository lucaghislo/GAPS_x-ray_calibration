import os.path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from distinctipy import distinctipy
from findpeaks import findpeaks

from read_events import *
from read_pedestals import *
from read_transfer_function import *
from calculate_xray_gain import *

# *** X-RAY DATA, PEDESTAL AND TRANSFER FUNCTION DATA ***
filepath_xray_data = r"input\xray_data\IT_400_xray_205_FTh_3mins_tau4.txt"
filepath_pedestal_data = r"input\pedestal_data\L4R0M0_Pedestals.dat"
filepath_fdt_data = r"input\transfer_function_data\L4R0M0_TransferFunction.dat"
folder_name = "IT_400_xray_205_FTh_3mins_tau4"

# Overwrite
filepath_xray_data = r"input\xray_data\xray_205_400_FTh_2mins.txt"
folder_name = "xray_205_400_FTh_2mins"

# *** CONFIGURATION ***
ch_min = 0
ch_max = 31
ASIC_number = 0
pt = 5
cadmium_peak = 88.0  # keV

# Maximum DAC_inj value for gain calculation in x-ray region
max_dac_inj_gain_linear = 200
max_dac_inj_gain_cubic = 500

channels = range(ch_min, ch_max + 1)
output_folder_path = os.path.join("output", folder_name)

# Plot configuration
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
matplotlib.rcParams["axes.linewidth"] = 0.7
matplotlib.rcParams["xtick.major.width"] = 0.7
matplotlib.rcParams["ytick.major.width"] = 0.7
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"


if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

# Events per channel organised in columns
print("** Reading events from file **")
events = read_events(filepath_xray_data, ASIC_number)

# Pedestal data
pedestals = read_pedestals(filepath_pedestal_data)

# Plot histogram of pedestal data per module
pedestal_folder = os.path.join(output_folder_path, "pedestal")

if not os.path.exists(pedestal_folder):
    os.mkdir(pedestal_folder)

plt.clf()
binwidth = 15
all_pedestals = pedestals[pedestals["pt"] == pt]
all_pedestals = all_pedestals["mean"]
(n, bins, patches) = plt.hist(
    all_pedestals,
    bins=range(
        int(min(all_pedestals)),
        int(max(all_pedestals)) + binwidth,
        binwidth,
    ),
    color="green",
)
plt.xlim(xmin=0, xmax=300)
plt.xlabel("Channel Output [ADU]")
plt.ylabel("Occurrences")
plt.title("\\textbf{Pedestal distribution for all channels}")

# Gaussian fit of data
(mu, sigma) = norm.fit(all_pedestals)

matplotlib.pyplot.text(
    10,
    max(n),
    "$\mu$ = "
    + str(round(mu, 2))
    + " ADU\n $\sigma$ = "
    + str(round(sigma, 2))
    + " ADU",
    fontsize=12,
    verticalalignment="top",
)

filename_ped = "allchs_pedestal_distribution.pdf"
plt.savefig(os.path.join(pedestal_folder, filename_ped))

# Save pedestal data: one file for all channels
filename_ped_data = "allch_pedestal_data.dat"
pedestal_data_file = os.path.join(pedestal_folder, filename_ped_data)
with open(
    pedestal_data_file,
    "w",
) as fp:
    count = 0
    for item in all_pedestals:
        fp.write(str(count) + "\t" + str(item) + "\n")
        count = count + 1

fp.close()

print("\n** Pedestals **")
print("Saved: " + filename_ped)
print("Saved: " + filename_ped_data)


# Transfer function data
fdt_data = read_transfer_function(filepath_fdt_data)

# Plot all transfer functions
fdt_folder = os.path.join(output_folder_path, "transfer_function")

if not os.path.exists(fdt_folder):
    os.mkdir(fdt_folder)

cal_v, out = get_fdt(fdt_data, 0, 0)
cal_v_kev = [cal_i * 0.841 for cal_i in cal_v]

colors = distinctipy.get_colors(len(channels))

# Make fdt single and data folder
single_fdt_folder = os.path.join(fdt_folder, "single")

if not os.path.exists(single_fdt_folder):
    os.mkdir(single_fdt_folder)

data_single_fdt_folder = os.path.join(single_fdt_folder, "data")

if not os.path.exists(data_single_fdt_folder):
    os.mkdir(data_single_fdt_folder)

plot_single_fdt_folder = os.path.join(single_fdt_folder, "plots")

if not os.path.exists(plot_single_fdt_folder):
    os.mkdir(plot_single_fdt_folder)

# Save single fdt plot
count = 0
for ch in channels:
    plt.clf()
    cal_v, out = get_fdt(fdt_data, ch, pt)
    plt.plot(cal_v_kev, out, colors[count])
    count = count + 1
    plt.xlabel("Incoming Energy [keV]")
    plt.ylabel("Channel Output [ADU]")
    plt.xlim(xmin=0, xmax=max(cal_v_kev))
    plt.ylim(ymin=0)
    plt.title(
        "\\textbf{Transfer function for channel " + str(ch) + "}", fontweight="bold"
    )
    filename_fdt_single = "fdt_ch" + str(ch) + "_pt" + str(pt) + ".pdf"
    plt.savefig(os.path.join(plot_single_fdt_folder, filename_fdt_single))

plt.clf()
count = 0
for ch in channels:
    cal_v, out = get_fdt(fdt_data, ch, pt)
    plt.plot(cal_v_kev, out, colors[count])
    count = count + 1

    filename_fdt_data = "fdt_ch" + str(ch) + "_pt" + str(pt) + ".dat"
    fdt_data_file = os.path.join(data_single_fdt_folder, filename_fdt_data)
    with open(
        fdt_data_file,
        "w",
    ) as fp:
        for i in range(0, len(cal_v)):
            fp.write(str(cal_v[i]) + "\t" + str(out[i]) + "\n")

    fp.close()

plt.xlabel("Incoming Energy [keV]")
plt.ylabel("Channel Output [ADU]")
plt.xlim(xmin=0, xmax=max(cal_v_kev))
plt.ylim(ymin=0)
plt.title("\\textbf{Transfer function for all channels}", fontweight="bold")

filename_fdt = "allchs_pt" + str(pt) + "_transfer_functions.pdf"
plt.savefig(os.path.join(fdt_folder, filename_fdt))

print("\n** Transfer functions **")
print("Saved: " + filename_fdt)
print("Saved: single channel fdt data in \\single\\data")
print("Saved: single channel fdt plot in \\single\\plots")


# Raw data histogram per channel
raw_main_folder = os.path.join(output_folder_path, "raw_data")

if not os.path.exists(raw_main_folder):
    os.mkdir(raw_main_folder)

raw_plot_folder = os.path.join(raw_main_folder, "plots")

if not os.path.exists(raw_plot_folder):
    os.mkdir(raw_plot_folder)

print("\n** Saving raw data plots **")
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    (n, bins, patches) = plt.hist(
        events_data,
        bins=range(
            int(min(events_data)),
            int(max(events_data)) + binwidth,
            binwidth,
        ),
        color="teal",
    )
    plt.xlim(xmin=0, xmax=300)
    plt.yscale("log")
    plt.xlabel("Energy [ADU]")
    plt.ylabel("occurrences")
    plt.title(
        "\\textbf{Raw data for channel " + str(ch) + " at tau " + str(pt) + "}",
        fontweight="bold",
    )

    filename_raw_data_plot = "ch" + str(ch) + "_" + "pt" + str(pt) + "_raw.pdf"
    raw_data_plot = os.path.join(raw_plot_folder, filename_raw_data_plot)
    plt.savefig(raw_data_plot)

    raw_data_folder = os.path.join(raw_main_folder, "data")

    if not os.path.exists(raw_data_folder):
        os.mkdir(raw_data_folder)

    # Write raw data to file
    filename_raw_data_file = "ch" + str(ch) + "_" + "pt" + str(pt) + "_raw.dat"
    raw_data_file = os.path.join(raw_data_folder, filename_raw_data_file)
    with open(
        raw_data_file,
        "w",
    ) as fp:
        for item in events_data:
            fp.write(str(item) + "\n")

    fp.close()

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_raw_data_plot))
    print("data: " + str(filename_raw_data_file) + "\n")


# Raw data histogram per channel with pedestal subtracted
raw_noped_main_folder = os.path.join(output_folder_path, "raw_no-pedestal_data")

if not os.path.exists(raw_noped_main_folder):
    os.mkdir(raw_noped_main_folder)

raw_noped_plot_folder = os.path.join(raw_noped_main_folder, "plots")

if not os.path.exists(raw_noped_plot_folder):
    os.mkdir(raw_noped_plot_folder)

print("** Saving raw data plots without pedestal **")
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    events_data_removed = [
        dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data
    ]
    (n, bins, patches) = plt.hist(
        events_data_removed,
        bins=range(
            int(min(events_data_removed)),
            int(max(events_data_removed)) + binwidth,
            binwidth,
        ),
        color="firebrick",
    )
    plt.xlim(xmin=0, xmax=300)
    plt.yscale("log")
    plt.xlabel("Energy [ADU]")
    plt.ylabel("occurrences")
    plt.title(
        "\\textbf{Raw data for channel "
        + str(ch)
        + " at tau "
        + str(pt)
        + " without pedestal}",
        fontweight="bold",
    )

    filename_raw_noped_data_plot = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_raw_no-pedestal.pdf"
    )
    raw_noped_data_plot = os.path.join(
        raw_noped_plot_folder,
        filename_raw_noped_data_plot,
    )
    plt.savefig(raw_noped_data_plot)

    raw_noped_data_folder = os.path.join(raw_noped_main_folder, "data")

    if not os.path.exists(raw_noped_data_folder):
        os.mkdir(raw_noped_data_folder)

    # Write raw data to file
    filename_raw_noped_data_file = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_raw_no-pedestal.dat"
    )
    raw_noped_data_file = os.path.join(
        raw_noped_data_folder,
        filename_raw_noped_data_file,
    )
    with open(
        raw_noped_data_file,
        "w",
    ) as fp:
        for item in events_data_removed:
            fp.write(str(item) + "\n")

    fp.close()

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_raw_noped_data_plot))
    print("data: " + str(filename_raw_noped_data_file) + "\n")


# LINEAR
# Calculate linear gain from interpolation for all selected channels
gain_folder = os.path.join(output_folder_path, "gain_x-ray_region")

if not os.path.exists(gain_folder):
    os.mkdir(gain_folder)

gain_folder_linear = os.path.join(gain_folder, "linear")

if not os.path.exists(gain_folder_linear):
    os.mkdir(gain_folder_linear)

gain_data_file_name = (
    "allchs_pt" + str(pt) + "_low_energy_gain_" + str(max_dac_inj_gain_linear) + ".dat"
)
gain_data_file = os.path.join(
    gain_folder_linear,
    gain_data_file_name,
)

gain_file = open(gain_data_file, "w")
gain_file.write("")
gain_file.close()
gain_file = open(gain_data_file, "a")

interpolation_folder = os.path.join(gain_folder_linear, "ch_interpolation")

if not os.path.exists(interpolation_folder):
    os.mkdir(interpolation_folder)

print("** Calculating linear gain for all channels in x-ray region **")
gains_lin = np.zeros(shape=(len(channels), 1))
pedestals_lin = np.zeros(shape=(len(channels), 1))
count = 0
for ch in channels:
    inter_filename = (
        "ch"
        + str(ch)
        + "_pt"
        + str(pt)
        + "_interp_"
        + str(max_dac_inj_gain_linear)
        + ".pdf"
    )
    inter_filepath = os.path.join(interpolation_folder, inter_filename)

    print("* Saved ch. " + str(ch) + " *")

    gain, pedestal = get_linear_gain(
        filepath_fdt_data, ch, pt, max_dac_inj_gain_linear, inter_filepath
    )
    gain_file.write(str(ch) + "\t" + str(gain) + "\t" + str(pedestal) + "\n")
    gains_lin[count] = gain
    pedestals_lin[count] = pedestal
    count = count + 1

gain_file.close()

# Plot gain per channel
plt.clf()
plt.plot(channels, gains_lin, marker="o")
plt.xlabel("Channel")
plt.ylabel("Gain [keV/ADU]")
plt.xticks(np.arange(min(channels), max(channels) + 1, step=5))
plt.title(
    "\\textbf{Linear gain up to " + str(max_dac_inj_gain_linear) + " DAC_inj_code}",
    fontweight="bold",
)
gain_trend_filename = "gain_chs_trend_" + str(max_dac_inj_gain_linear) + ".pdf"
plt.savefig(os.path.join(gain_folder_linear, gain_trend_filename))

# Plot histogram of pedestals obtained from linear interpolation
plt.clf()
binwidth = 15
all_pedestals = pedestals_lin
(n, bins, patches) = plt.hist(
    pedestals_lin,
    bins=range(
        int(min(all_pedestals)),
        int(max(all_pedestals)) + binwidth,
        binwidth,
    ),
    color="dodgerblue",
)
plt.xlim(xmin=0, xmax=300)
plt.xlabel("Channel Output [ADU]")
plt.ylabel("Occurrences")
plt.title("\\textbf{Estimated pedestals for all channels}", fontweight="bold")

# Gaussian fit of data
(mu, sigma) = norm.fit(all_pedestals)

matplotlib.pyplot.text(
    10,
    max(n),
    "$\mu$ = "
    + str(round(mu, 2))
    + " ADU\n $\sigma$ = "
    + str(round(sigma, 2))
    + " ADU",
    fontsize=12,
    verticalalignment="top",
)

filename_ped = "allchs_estimated_pedestal_distribution.pdf"
plt.savefig(os.path.join(gain_folder_linear, filename_ped))

print("Saved: " + gain_data_file_name)
print("Saved: " + gain_trend_filename)
print("Saved: " + filename_ped)


# CUBIC
# Calculate cubic gain from interpolation for all selected channels
gain_folder_cubic = os.path.join(gain_folder, "cubic")

if not os.path.exists(gain_folder_cubic):
    os.mkdir(gain_folder_cubic)

gain_data_file_name = (
    "allchs_pt" + str(pt) + "_low_energy_gain_" + str(max_dac_inj_gain_cubic) + ".dat"
)
gain_data_file = os.path.join(
    gain_folder_cubic,
    gain_data_file_name,
)

gain_file = open(gain_data_file, "w")
gain_file.write("")
gain_file.close()
gain_file = open(gain_data_file, "a")

interpolation_folder = os.path.join(gain_folder_cubic, "ch_interpolation")

if not os.path.exists(interpolation_folder):
    os.mkdir(interpolation_folder)

print("** Calculating cubic gain for all channels in x-ray region **")
gains_cubic = np.zeros(shape=(len(channels), 1))
pedestals_cubic = np.zeros(shape=(len(channels), 1))
count = 0
for ch in channels:
    inter_filename = (
        "ch"
        + str(ch)
        + "_pt"
        + str(pt)
        + "_interp_"
        + str(max_dac_inj_gain_cubic)
        + ".pdf"
    )
    inter_filepath = os.path.join(interpolation_folder, inter_filename)

    print("* Saved ch. " + str(ch) + " *")

    gain, pedestal = get_cubic_gain(
        filepath_fdt_data, ch, pt, max_dac_inj_gain_cubic, inter_filepath
    )
    gain_file.write(str(ch) + "\t" + str(gain) + "\t" + str(pedestal) + "\n")
    gains_cubic[count] = gain
    pedestals_cubic[count] = pedestal
    count = count + 1

gain_file.close()

# Plot gain per channel
plt.clf()
plt.plot(channels, gains_cubic, marker="o")
plt.xlabel("Channel")
plt.ylabel("Gain [keV/ADU]")
plt.xticks(np.arange(min(channels), max(channels) + 1, step=5))
plt.title(
    "\\textbf{Cubic gain up to " + str(max_dac_inj_gain_cubic) + " DAC_inj_code}",
    fontweight="bold",
)
gain_trend_filename = "gain_chs_trend_" + str(max_dac_inj_gain_cubic) + ".pdf"
plt.savefig(os.path.join(gain_folder_cubic, gain_trend_filename))

# Plot histogram of pedestals obtained from cubic interpolation
plt.clf()
binwidth = 15
all_pedestals = pedestals_cubic
(n, bins, patches) = plt.hist(
    pedestals_cubic,
    bins=range(
        int(min(all_pedestals)),
        int(max(all_pedestals)) + binwidth,
        binwidth,
    ),
    color="dodgerblue",
)
plt.xlim(xmin=0, xmax=300)
plt.xlabel("Channel Output [ADU]")
plt.ylabel("Occurrences")
plt.title("\\textbf{Estimated pedestals for all channels}", fontweight="bold")

# Gaussian fit of data
(mu, sigma) = norm.fit(all_pedestals)

matplotlib.pyplot.text(
    10,
    max(n),
    "$\mu$ = "
    + str(round(mu, 2))
    + " ADU\n $\sigma$ = "
    + str(round(sigma, 2))
    + " ADU",
    fontsize=12,
    verticalalignment="top",
)

filename_ped = "allchs_estimated_pedestal_distribution.pdf"
plt.savefig(os.path.join(gain_folder_cubic, filename_ped))

print("Saved: " + gain_data_file_name)
print("Saved: " + gain_trend_filename)
print("Saved: " + filename_ped)

# ADU -> keV conversion and plot
# Linear gain
converted_noped_main_folder = os.path.join(
    output_folder_path, "converted_no-pedestal_data"
)

if not os.path.exists(converted_noped_main_folder):
    os.mkdir(converted_noped_main_folder)

converted_noped_main_folder_linear = os.path.join(converted_noped_main_folder, "linear")

if not os.path.exists(converted_noped_main_folder_linear):
    os.mkdir(converted_noped_main_folder_linear)

converted_noped_plot_folder = os.path.join(converted_noped_main_folder_linear, "plots")

if not os.path.exists(converted_noped_plot_folder):
    os.mkdir(converted_noped_plot_folder)

print("\n** Saving converted data plots without pedestal **")
plt.clf()
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    events_data_removed = [
        dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data
    ]
    gain, pedestal = get_linear_gain(filepath_fdt_data, ch, pt, max_dac_inj_gain_linear)
    events_data_removed_kev = [dat_i * gain for dat_i in events_data_removed]
    (n, bins, patches) = plt.hist(
        events_data_removed_kev,
        bins=range(
            int(min(events_data_removed_kev)),
            int(max(events_data_removed_kev)) + binwidth,
            binwidth,
        ),
        color="purple",
    )
    plt.xlim(xmin=0, xmax=300)
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("occurrences")
    plt.title(
        "\\textbf{Converted data for channel "
        + str(ch)
        + " at tau "
        + str(pt)
        + " without pedestal}",
        fontweight="bold",
    )

    filename_converted_noped_data_plot = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_keV_no-pedestal.pdf"
    )
    converted_noped_data_plot = os.path.join(
        converted_noped_plot_folder,
        filename_converted_noped_data_plot,
    )
    plt.savefig(converted_noped_data_plot)

    converted_noped_data_folder = os.path.join(
        converted_noped_main_folder_linear, "data"
    )

    if not os.path.exists(converted_noped_data_folder):
        os.mkdir(converted_noped_data_folder)

    # Write converted data to file
    filename_converted_noped_data_file = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_kev_no-pedestal.dat"
    )
    converted_noped_data_file = os.path.join(
        converted_noped_data_folder,
        filename_converted_noped_data_file,
    )
    with open(
        converted_noped_data_file,
        "w",
    ) as fp:
        for item in events_data_removed_kev:
            fp.write(str(item) + "\n")

    fp.close()

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_converted_noped_data_plot))
    print("data: " + str(filename_converted_noped_data_file) + "\n")


# ADU -> keV conversion and plot
# Cubic gain
converted_noped_main_folder_cubic = os.path.join(converted_noped_main_folder, "cubic")

if not os.path.exists(converted_noped_main_folder_cubic):
    os.mkdir(converted_noped_main_folder_cubic)

converted_noped_plot_folder = os.path.join(converted_noped_main_folder_cubic, "plots")

if not os.path.exists(converted_noped_plot_folder):
    os.mkdir(converted_noped_plot_folder)

print("\n** Saving converted data plots without pedestal **")
plt.clf()
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    events_data_removed = [
        dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data
    ]
    gain, pedestal = get_cubic_gain(filepath_fdt_data, ch, pt, max_dac_inj_gain_cubic)
    events_data_removed_kev = [dat_i * gain for dat_i in events_data_removed]
    (n, bins, patches) = plt.hist(
        events_data_removed_kev,
        bins=range(
            int(min(events_data_removed_kev)),
            int(max(events_data_removed_kev)) + binwidth,
            binwidth,
        ),
        color="blue",
    )
    plt.xlim(xmin=0, xmax=300)
    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("occurrences")
    plt.title(
        "\\textbf{Converted data for channel "
        + str(ch)
        + " at tau "
        + str(pt)
        + " without pedestal}",
        fontweight="bold",
    )

    filename_converted_noped_data_plot = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_keV_no-pedestal.pdf"
    )
    converted_noped_data_plot = os.path.join(
        converted_noped_plot_folder,
        filename_converted_noped_data_plot,
    )
    plt.savefig(converted_noped_data_plot)

    converted_noped_data_folder = os.path.join(
        converted_noped_main_folder_cubic, "data"
    )

    if not os.path.exists(converted_noped_data_folder):
        os.mkdir(converted_noped_data_folder)

    events_data_removed_kev_lim = [
        events_data_removed_kev[i]
        for i in range(len(events_data_removed_kev))
        if (events_data_removed_kev[i] >= 70) and (events_data_removed_kev[i] <= 100)
    ]

    # Cadmium peak
    cadmium_peak_folder = os.path.join(output_folder_path, "cadmium_peak")

    if not os.path.exists(cadmium_peak_folder):
        os.mkdir(cadmium_peak_folder)

    events_data = get_events(events, ch)
    events_data_removed = [
        dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data
    ]

    events_data_removed_lim = [
        events_data_removed[i]
        for i in range(len(events_data_removed))
        if (events_data_removed[i] >= 60) and (events_data_removed[i] <= 100)
    ]

    binwidth = 1
    plt.clf()
    (n1, bins, patches) = plt.hist(
        events_data_removed_kev_lim,
        bins=np.arange(
            int(min(events_data_removed_kev_lim)),
            int(max(events_data_removed_kev_lim)) + binwidth,
            binwidth,
        ),
        label="keV",
        alpha=1,
    )

    max1 = max(n1)

    # (n2, bins, patches) = plt.hist(
    #     events_data_removed_lim,
    #     bins=np.arange(
    #         int(min(events_data_removed)),
    #         int(max(events_data_removed)) + binwidth,
    #         binwidth,
    #     ),
    #     label="ADU",
    #     alpha=0.5,
    # )

    # max2 = max(n2)

    plt.xlim(30, 120)
    plt.ylim(0, max1 + 2)
    # plt.title("\\textbf{Cadmium peak comparison before/after conversion}")
    plt.title("\\textbf{Cadmium peak after conversion}")
    plt.xlabel("Incoming energy [keV]")
    plt.ylabel("Occurrences")
    # plt.legend()
    plt.savefig(os.path.join(cadmium_peak_folder, "peak_ch" + str(ch) + ".pdf"))

    # Write converted data to file
    filename_converted_noped_data_file = (
        "ch" + str(ch) + "_" + "pt" + str(pt) + "_kev_no-pedestal.dat"
    )
    converted_noped_data_file = os.path.join(
        converted_noped_data_folder,
        filename_converted_noped_data_file,
    )
    with open(
        converted_noped_data_file,
        "w",
    ) as fp:
        for item in events_data_removed_kev:
            fp.write(str(item) + "\n")

    fp.close()

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_converted_noped_data_plot))
    print("data: " + str(filename_converted_noped_data_file) + "\n")
