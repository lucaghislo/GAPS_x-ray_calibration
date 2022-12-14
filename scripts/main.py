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
ASIC_number = 0

# Events per channel organised in columns
events = read_events(filepath_xray_data, ASIC_number)

# Pedestal data
pedestals = read_pedestals(filepath_pedestal_data)


# # Events histogram of all channels
# plt.clf()
# binwidth = 1
# plt.hist(
#     events_data,
#     bins=range(int(min(events_data)), int(max(events_data)) + binwidth, binwidth),
# )
# plt.xlim(xmin=0, xmax=300)
# plt.ylim(ymin=0, ymax=35000)
# plt.show()

# # Save raw data histograms
# for ch in range(ch_min, ch_max):

#     # ch_events = good_events[good_events[]]

#     plt.clf()
#     binwidth = 1
#     plt.hist(
#         events_data,
#         bins=range(int(min(events_data)), int(max(events_data)) + binwidth, binwidth),
#     )
#     plt.xlim(xmin=0, xmax=300)
#     plt.ylim(ymin=0, ymax=35000)


# # Read pedestals
# pedestals_raw = pd.read_csv("pedestal_data\L4R0M0_Pedestals.dat")
