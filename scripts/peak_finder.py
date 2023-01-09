import os.path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from distinctipy import distinctipy
from findpeaks import findpeaks
from scipy import signal
from scipy.signal import savgol_filter

from read_events import *
from read_pedestals import *
from read_transfer_function import *
from calculate_xray_gain import *

ch = 0
max_indexes = np.zeros(shape=(32, 1))

# data = pd.read_csv(
#     r"output\xray_205_400_FTh_2mins\converted_no-pedestal_data\cubic\data\ch0_pt5_kev_no-pedestal.dat",
#     sep="\t",
# )

for ch in range(0, 32):
    data = pd.read_csv(
        r"output\IT_400_xray_205_FTh_3mins_tau4_2\raw_no-pedestal_data\data\ch"
        + str(ch)
        + r"_pt4_raw_no-pedestal.dat",
        sep="\t",
    )

    # print(data)

    # data = data[(data >= 60) & (data <= 100)]
    data = data[(data >= 0) & (data <= 300)]

    plt.clf()
    (n, bins, h) = plt.hist(data, bins=150)
    # plt.clf()
    # plt.xlim([0, 300])
    plt.yscale("log")

    x = bins[0 : len(bins) - 1]

    yhat = savgol_filter(n, 4, 3)
    yhat = n
    plt.plot(x, yhat, linewidth=0, marker="*")

    max_peak_hat = 0
    max_peak_hat_index = 0
    for i in range(0, len(yhat)):
        if yhat[i] > max_peak_hat and x[i] > 60:
            max_peak_hat = yhat[i]
            max_peak_hat_index = x[i]

    print("--> " + str(max_peak_hat_index))

    # print(n)
    # print(bins)
    # print(len(bins))

    max_peak_width = 5
    y_coordinates = np.array(
        n
    )  # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    peak_widths = np.arange(1, max_peak_width)
    peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
    peak_count = len(peak_indices)  # the number of peaks in the array

    # print(peak_count)
    # print("")
    # print(peak_indices)

    max_peak = 0
    max_ind = 0
    peak_vals = np.zeros(shape=(32, 2))
    for i in range(0, len(peak_indices)):
        if x[peak_indices[i]] >= 75 and x[peak_indices[i]] <= 120:
            # print("Picco: " + str(x[peak_indices[i]]) + "\t" + str(n[peak_indices[i]]))
            peak_vals[i, 0] = x[peak_indices[i]]
            peak_vals[i, 1] = n[peak_indices[i]]

            if n[peak_indices[i]] > max_peak:
                max_peak = n[peak_indices[i]]
                max_ind = x[peak_indices[i]]

    print("Ch. " + str(ch) + ", max: " + str(max_peak) + " @ " + str(max_ind) + " ADU")
    max_indexes[ch] = max_ind  # max_ind

    # plt.plot(x[peak_indices], n[peak_indices], linewidth=1, marker="o")
    plt.plot(peak_vals[:, 0], peak_vals[:, 1], linewidth=0, marker="o")
    plt.xticks(range(-50, 300, 10))
    # plt.show()

    # print(x[peak_indices])

gains_data = pd.read_csv(
    # r"output\xray_205_400_FTh_2mins\gain_x-ray_region\cubic\allchs_pt5_low_energy_gain_500.dat",
    # r"output\xray_205_400_FTh_2mins\gain_x-ray_region\linear\allchs_pt5_low_energy_gain_200.dat",
    r"output\IT_400_xray_205_FTh_3mins_tau4_2\gain_x-ray_region\cubic\allchs_pt4_low_energy_gain_500.dat",
    sep="\t",
    header=None,
)

gains = gains_data.iloc[:, 1]

print("Lunghezza guadagni: " + str(len(gains)))

print("Guadagni:")
print(gains)

print("Picchi ADU")
peaks_adu = max_indexes
print(peaks_adu)
print("\n")

peaks_kev = np.zeros(shape=(32, 1))
for i in range(0, 32):
    peak_kev = peaks_adu[i] * gains[i]
    print("Ch. " + str(i) + ": " + str(peaks_adu[i]) + " -> " + str(peak_kev))
    peaks_kev[i] = peak_kev

plt.clf()
peaks_adu = peaks_adu[peaks_adu > 0]
peaks_kev = peaks_kev[peaks_kev > 0]
plt.hist(peaks_adu, alpha=0.5)
plt.hist(peaks_kev, alpha=0.5)

print(str(np.mean(peaks_adu)) + "\t" + str(np.std(peaks_adu)))
print(str(np.mean(peak_kev)) + "\t" + str(np.std(peaks_kev)))
print(np.std(gains))
plt.show()
