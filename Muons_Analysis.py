import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = "/Users/zeke/PycharmProjects/MPL_Muons/RawMuonData.xlsx"


def import_excel(pth):
    """
    Imports Muon Data from Excel into Python using Panda
    :param pth: File path of Excel Data
    :return: Time duration for Muon Pairs
    """

    data = pd.read_excel(pth)
    time_data = pd.DataFrame(data, columns=['Time'])    # In milliseconds
    return data


def pd_2_np(data):
    np_array = data.to_numpy()
    return np_array


def print_hist(pth):
    data = import_excel(pth)
    histo = data.hist()
    print(histo)


def theory_model(x, *params):
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    decay_model = p1*np.exp(-x/p2) + p3

    return decay_model


def hist_plot(pth, bins=40):

    data = pd_2_np(import_excel(pth))
    y, binEdges = np.histogram(data, bins=np.arange(0, bins+1), range=20)

    bin_centers = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 0.75
    menStd = np.sqrt(y)

    sigma = [np.sqrt(i) for i in y]
    p0 = np.array([1000, 0.5, 150])
    x_data = 0.5 * (binEdges[1:] + binEdges[:-1])

    p, cov = opt.curve_fit(theory_model, x_data, y, p0, sigma)
    new_model = theory_model(x_data, *p)

    # Graphing of data
    plt.bar(
        bin_centers,
        y,
        width=width,
        color='b',
        label="Hist")

    plt.plot(
        bin_centers,
        new_model,
        "k--",
        label="Fit",
        color="r")
    plt.errorbar(
        bin_centers,
        y,
        yerr=np.sqrt(y),
        fmt='None',    # Prevents graphing of line
        ecolor='black',
        marker='.'
    )
    plt.xticks(np.arange(0, bins+1))
    plt.legend(loc="upper right")
    plt.show()




# print_hist(path)
hist_plot(path)

