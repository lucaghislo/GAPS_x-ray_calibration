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

# *** X-RAY DATA, PEDESTAL AND TRANSFER FUNCTION DATA ***
filepath_xray_data = "xray_data\IT_400_xray_205_FTh_3mins_tau4.txt"
filepath_pedestal_data = "pedestal_data\L4R0M0_Pedestals.dat"
filepath_fdt_data = "transfer_function_data\L4R0M0_TransferFunction.dat"
folder_name = "IT_400_xray_205_FTh_3mins_tau4"

# *** CONFIGURATION ***
ch_min = 0
ch_max = 31
ASIC_number = 0
pt = 4

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

print("\n** Pedestals **")
print("Saved: " + filename_ped)

# Transfer function data
fdt_data = read_transfer_function(filepath_fdt_data)

# Plot all transfer functions
fdt_folder = os.path.join(output_folder_path, "transfer_function")

if not os.path.exists(fdt_folder):
    os.mkdir(fdt_folder)

cal_v, out = get_fdt(fdt_data, 0, 0)
cal_v_kev = [cal_i * 0.841 for cal_i in cal_v]

colors = distinctipy.get_colors(len(channels))

plt.clf()
count = 0
for ch in channels:
    cal_v, out = get_fdt(fdt_data, ch, pt)
    plt.plot(cal_v_kev, out, colors[count])
    count = count + 1

plt.xlabel("Incoming Energy [keV]")
plt.ylabel("Channel Output [ADU]")
plt.xlim(xmin=0, xmax=max(cal_v_kev))
plt.ylim(ymin=0)
plt.title("Transfer function for all channels", fontweight="bold")

filename_fdt = "allchs_transfer_functions.pdf"
plt.savefig(os.path.join(fdt_folder, filename_fdt))

print("\n** Transfer functions **")
print("Saved: " + filename_fdt)

# Raw data histogram per channel
raw_main_folder = os.path.join(output_folder_path, "raw_data")

if not os.path.exists(raw_main_folder):
    os.mkdir(raw_main_folder)

raw_plot_folder = os.path.join(raw_main_folder, "plots")

if not os.path.exists(raw_plot_folder):
    os.mkdir(raw_plot_folder)

print("\n**Saving raw data plots**\n")
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

    print("*Saved ch. " + str(ch) + "*")
    print("plot: " + str(filename_raw_data_plot))
    print("data: " + str(filename_raw_data_file) + "\n")

# Raw data histogram per channel with pedestal subtracted
raw_noped_main_folder = os.path.join(output_folder_path, "raw_no-pedestal_data")

if not os.path.exists(raw_noped_main_folder):
    os.mkdir(raw_noped_main_folder)

raw_noped_plot_folder = os.path.join(raw_noped_main_folder, "plots")

if not os.path.exists(raw_noped_plot_folder):
    os.mkdir(raw_noped_plot_folder)

print("\n**Saving raw data plots without pedestal**\n")
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

    print("*Saved ch. " + str(ch) + "*")
    print("plot: " + str(filename_raw_noped_data_plot))
    print("data: " + str(filename_raw_noped_data_file) + "\n")
