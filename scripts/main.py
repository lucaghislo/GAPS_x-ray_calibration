import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from read_events import *
from read_pedestals import *

filepath_xray_data = "xray_data\IT_400_xray_205_FTh_3mins_tau4.txt"
filepath_pedestal_data = "pedestal_data\L4R0M0_Pedestals.dat"
ch_min = 0
ch_max = 31
channels = range(ch_min, ch_max + 1)
ASIC_number = 0
pt = 4

# Events per channel organised in columns
events = read_events(filepath_xray_data, ASIC_number)

# Pedestal data
pedestals = read_pedestals(filepath_pedestal_data)

# Raw data histogram per channel
print("\n***Saving raw data plots***")
for ch in channels:
    plt.clf()
    binwidth = 1
    events_data = get_events(events, ch)
    events_data = [dat_i - get_pedestal(pedestals, ch, pt) for dat_i in events_data]
    (n, bins, patches) = plt.hist(
        events_data,
        bins=range(int(min(events_data)), int(max(events_data)) + binwidth, binwidth),
    )
    plt.xlim(xmin=0, xmax=300)
    plt.yscale("log")
    plt.title("Raw data for channel " + str(ch) + "at tau " + str(pt))
    plt.savefig(
        r"output\raw_data_no-pedestal\ch"
        + str(ch)
        + "_"
        + "pt"
        + str(pt)
        + "_no-ped.pdf"
    )
    print("Saved ch. " + str(ch))
