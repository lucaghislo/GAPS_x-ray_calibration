import os.path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from distinctipy import distinctipy

from read_events import *
from read_pedestals import *
from read_transfer_function import *
from calculate_xray_gain import *

# *** X-RAY DATA, PEDESTAL AND TRANSFER FUNCTION DATA ***
filepath_xray_data = r"input\xray_data\IT_400_xray_205_FTh_3mins_tau4.txt"
filepath_pedestal_data = r"input\pedestal_data\L4R0M0_Pedestals.dat"
filepath_fdt_data = r"input\transfer_function_data\L4R0M0_TransferFunction.dat"
folder_name = "IT_400_xray_205_FTh_3mins_tau4"

# *** CONFIGURATION ***
ch_min = 0
ch_max = 31
ASIC_number = 0
pt = 4

# Maximum DAC_inj value for linear gain calculation in x-ray region
max_dac_inj_gain = 100

channels = range(ch_min, ch_max + 1)
output_folder_path = os.path.join("output", folder_name)

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
plt.title("Pedestal distribution for all channels", fontweight="bold")

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
    plt.title("Transfer function for channel " + str(ch), fontweight="bold")
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

plt.xlabel("Incoming Energy [keV]")
plt.ylabel("Channel Output [ADU]")
plt.xlim(xmin=0, xmax=max(cal_v_kev))
plt.ylim(ymin=0)
plt.title("Transfer function for all channels", fontweight="bold")

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
    plt.ylabel("Occurences")
    plt.title(
        "Raw data for channel " + str(ch) + " at tau " + str(pt),
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
    plt.ylabel("Occurences")
    plt.title(
        "Raw data for channel " + str(ch) + " at tau " + str(pt) + " without pedestal",
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
        for item in events_data:
            fp.write(str(item) + "\n")

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_raw_noped_data_plot))
    print("data: " + str(filename_raw_noped_data_file) + "\n")


# Calculate gain from interpolation for all selected channels
gain_folder = os.path.join(output_folder_path, "gain_x-ray_region")

if not os.path.exists(gain_folder):
    os.mkdir(gain_folder)

gain_data_file_name = (
    "allchs_pt" + str(pt) + "_low_energy_gain_" + str(max_dac_inj_gain) + ".dat"
)
gain_data_file = os.path.join(
    gain_folder,
    gain_data_file_name,
)

gain_file = open(gain_data_file, "w")
gain_file.write("")
gain_file = open(gain_data_file, "a")

gains_lin = np.zeros(shape=(len(channels), 1))
pedestals_lin = np.zeros(shape=(len(channels), 1))
count = 0
for ch in channels:
    gain, pedestal = get_linear_gain(filepath_fdt_data, ch, pt, max_dac_inj_gain)
    gain_file.write(str(ch) + "\t" + str(gain) + "\t" + str(pedestal) + "\n")
    gains_lin[count] = gain
    pedestals_lin[count] = pedestal
    count = count + 1

# Plot gain per channel
plt.clf()
plt.plot(channels, gains_lin, marker="o", linestyle="None")
plt.xlabel("Channel")
plt.ylabel("Gain [keV/ADU]")
plt.xticks(np.arange(min(channels), max(channels) + 1, step=1))
plt.title(
    "Linear gain up to " + str(max_dac_inj_gain) + " DAC_inj_code", fontweight="bold"
)
gain_trend_filename = "gain_chs_trend_" + str(max_dac_inj_gain) + ".pdf"
plt.savefig(os.path.join(gain_folder, gain_trend_filename))


# Plot histogram of pedestals obtained from linear interpolation
# Plot histogram of pedestal data per module
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
plt.title("Estimated pedestals for all channels", fontweight="bold")

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
plt.savefig(os.path.join(gain_folder, filename_ped))

print("** Calculated linear gain for all channels in x-ray region **")
print("Saved: " + gain_data_file_name)
print("Saved: " + gain_trend_filename)
print("Saved: " + filename_ped)


# ADU -> keV conversion and plot
converted_noped_main_folder = os.path.join(
    output_folder_path, "converted_no-pedestal_data"
)

if not os.path.exists(converted_noped_main_folder):
    os.mkdir(converted_noped_main_folder)

converted_noped_plot_folder = os.path.join(converted_noped_main_folder, "plots")

if not os.path.exists(converted_noped_plot_folder):
    os.mkdir(converted_noped_plot_folder)

print("\n** Saving converted data plots without pedestal **")
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    events_data_removed = [
        dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data
    ]
    gain, pedestal = get_linear_gain(filepath_fdt_data, ch, pt, max_dac_inj_gain)
    events_data_removed_kev = [dat_i * gain for dat_i in events_data_removed]
    (n, bins, patches) = plt.hist(
        events_data_removed,
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
    plt.ylabel("Occurences")
    plt.title(
        "Converted data for channel "
        + str(ch)
        + " at tau "
        + str(pt)
        + " without pedestal",
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

    converted_noped_data_folder = os.path.join(converted_noped_main_folder, "data")

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
        for item in events_data:
            fp.write(str(item) + "\n")

    print("* Saved ch. " + str(ch) + " *")
    print("plot: " + str(filename_converted_noped_data_plot))
    print("data: " + str(filename_converted_noped_data_file) + "\n")
