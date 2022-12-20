import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from read_events import *
from read_pedestals import *

# Filepaths for x-ray and pedestal
filepath_xray_data = "xray_data\IT_400_xray_205_FTh_3mins_tau4.txt"
filepath_pedestal_data = "pedestal_data\L4R0M0_Pedestals.dat"
folder_name = "IT_400_xray_205_FTh_3mins_tau4"

# Configuration data
ch_min = 0
ch_max = 31
ASIC_number = 0
pt = 4

channels = range(ch_min, ch_max + 1)
output_folder_path = os.path.join("output", folder_name)

if not os.path.exists(output_folder_path):
    os.mkdir(output_folder_path)

# Events per channel organised in columns
events = read_events(filepath_xray_data, ASIC_number)

# Pedestal data
pedestals = read_pedestals(filepath_pedestal_data)

# Raw data histogram per channel
raw_main_folder = os.path.join(output_folder_path, "raw_data")

if not os.path.exists(raw_main_folder):
    os.mkdir(raw_main_folder)

raw_plot_folder = os.path.join(raw_main_folder, "plots")

if not os.path.exists(raw_plot_folder):
    os.mkdir(raw_plot_folder)

print("\n***Saving raw data plots***\n")
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

print("\n***Saving raw data plots without pedestal***\n")
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
