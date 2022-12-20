import os.path
import pandas as pd
import numpy as np


def read_events(filepath, ASIC_number):
    # Get only real events (type = 00 or type = 10)
    ev, evtype, module, chtype, channel, adc = np.loadtxt(
        filepath,
        dtype="int",
        comments="#",
        usecols=(0, 1, 2, 3, 4, 5),
        unpack=True,
    )
    N = np.unique(ev)

    print("         Events in file: ", len(N))
    Event = []

    idx = (module == ASIC_number).nonzero()[0]
    evn = ev[idx]
    ch = channel[idx]
    val = adc[idx]

    evnw = evn.view()
    chw = ch.view()
    valw = val.view()

    selected = 0
    all_neg_saturation = 0
    all_pos_saturation = 0
    some_neg_saturation = 0
    some_pos_saturation = 0
    neg_saturation = 40.0
    pos_saturation = 2020.0

    if len(N):
        for n in N:
            # check only 40 rows at a time
            if len(evnw) > 40:
                idx = (evnw[0:40] == n).nonzero()[0]
            else:
                idx = (evnw == n).nonzero()[0]
            # verify that the event contains 32 rows and all channels are present
            if (len(idx) == 32) and (ch[idx].sum() == 496):
                # verify saturation
                if (valw[idx] <= neg_saturation).all():
                    all_neg_saturation += 1
                elif (valw[idx] <= neg_saturation).any():
                    some_neg_saturation += 1
                elif (valw[idx] >= pos_saturation).all():
                    all_pos_saturation += 1
                elif (valw[idx] >= pos_saturation).any():
                    some_pos_saturation += 1
                else:
                    Nch = chw[idx]
                    ADCs = valw[idx]
                    Event.append([n, Nch, ADCs])
                    selected += 1

            ii = len(idx)
            evnw = evnw[ii:].view()
            chw = chw[ii:].view()
            valw = valw[ii:].view()

    # Plot the good/bad events statistics
    print(
        "            Real events: ",
        selected
        + all_neg_saturation
        + some_neg_saturation
        + all_pos_saturation
        + some_pos_saturation,
    )
    print(" All negative saturated: ", all_neg_saturation)
    print("Some negative saturated: ", some_neg_saturation)
    print(" All positive saturated: ", all_pos_saturation)
    print("Some positive saturated: ", some_pos_saturation)
    print("               Selected: ", selected)

    # events per channel listed by column
    global events
    global peds
    n = len(Event)
    peds = np.zeros((1, 32))
    events = np.empty((n, 32))
    for i, data in enumerate(Event):
        events[i, :] = data[2]

    return events


def get_events(events, ch):
    return events[:, ch]
