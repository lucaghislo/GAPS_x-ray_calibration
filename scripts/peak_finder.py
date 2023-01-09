import os.path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from distinctipy import distinctipy
from findpeaks import findpeaks
from scipy import signal

from read_events import *
from read_pedestals import *
from read_transfer_function import *
from calculate_xray_gain import *

data = pd.read_csv(
    r"output\xray_205_400_FTh_2mins\converted_no-pedestal_data\cubic\data\ch0_pt5_kev_no-pedestal.dat",
    sep="\t",
)

# print(data)

data = data[data <= 300]

plt.clf()
(n, bins, h) = plt.hist(data, bins=300)
plt.clf()
plt.xlim([0, 300])
plt.yscale("log")

x = bins[0 : len(bins) - 1]

plt.plot(x, n)

# print(n)
print(bins)
print(len(bins))

max_peak_width = 2
y_coordinates = np.array(
    n
)  # convert your 1-D array to a numpy array if it's not, otherwise omit this line
peak_widths = np.arange(1, max_peak_width)
peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
peak_count = len(peak_indices)  # the number of peaks in the array

print(peak_count)
print("")
print(peak_indices)

plt.plot(x[peak_indices], n[peak_indices], linewidth=0, marker="o")
plt.show()

print(x[peak_indices])
