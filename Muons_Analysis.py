import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy import optimize as opt
from tabulate import tabulate


def import_excel(pth):
    """
    Imports Muon Data from Excel into Python using Panda
    Converts to NumPy Array
    :param pth: File path of Excel Data
    :return: NumPy Array containing list of time duration values for muon pairs
    """

    data = pd.read_excel(pth)
    time_data = pd.DataFrame(data, columns=['Time'])    # In milliseconds
    np_array = time_data.to_numpy()    # Convert from Panda DataFrame to NumPy Array

    return np_array


def theory_model(x, *params):
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]

    model = p1*np.exp(-x/p2) + p3

    return model


def make_json(runs, bins, trim):
    l_runs, l_bins = len(runs), len(bins)
    init_dict = {
        "bins": bins,
        "trim": trim
    }

    for i in range(l_runs):
        init_dict[f"Run {i+1}"] = {
            "total": 0,
            "chi squared": [],
            "chi reduced": [],
            "tau": [],
            "tau unc": [],
            "n0": [],
            "background": [],
            "raw_data_file": runs[i],
            "plot_file": []
        }

    with open("muon_data_analysis.json", 'w+') as file:
        json.dump(init_dict, file, indent=3)


def store_data(counts, params, cov, run, pltfile, chisq, chi_red):

    # Retrieves JSON file to be written to
    with open("muon_data_analysis.json", 'r') as file:
        datafile = json.load(file)

    # Writes to JSON file
    with open("muon_data_analysis.json", 'w+') as file:
        if datafile[f"Run {run}"]["total"] == 0:
            datafile[f"Run {run}"]["total"] = counts

        datafile[f"Run {run}"]["chi squared"].append(chisq)
        datafile[f"Run {run}"]["chi reduced"].append(chi_red)
        datafile[f"Run {run}"]["tau"].append(params[1])
        datafile[f"Run {run}"]["tau unc"].append(cov[1][1])
        datafile[f"Run {run}"]["n0"].append(params[0])
        datafile[f"Run {run}"]["background"].append(params[2])
        datafile[f"Run {run}"]["plot_file"].append(pltfile)

        json.dump(datafile, file, indent=3)    # Saves new dict to JSON


# Helper Function
def tbl_data(x, y, sig, thr, run, step):
    """
    Prints table of data to console. Good for debugging purposes.

    :param x: [Column 1] Represents x data... in this case time bins
              Advised to set to 'bin_edges' or 'bin_centers'
    :param y: [Column 2] Counts (Observed y)
    :param sig: [Col 3] Uncertainty of y
    :param thr: [Col 4] Output from model using best fit params (Expected y)
    :param run: Current Data Run
    :param step: Bin widths in microseconds
    :return: None
    """

    print(f"Run {run} Binning {step}\n")
    headers = ["Bin Centers", "Counts", "Uncertainty", "Theory"]
    table = zip(x, y, sig, thr)
    print(f"{tabulate(table, headers=headers, floatfmt='.4f')}\n")


def plot_data(params):
    """
    # plot_input = [bin_centers, y, step, new_model, max_val, run, bins]
    :param params:
    :return:
    """

    # Graphing of histogram data w/ error
    plt.bar(params[0], params[1], width=params[2], color='b', label="Hist")
    plt.errorbar(
        params[0],
        params[1],
        yerr=np.sqrt(params[1]),
        fmt='None',  # Prevents graphing of line
        ecolor='black',
        marker='.'
    )
    # Plot line of best fit
    plt.plot(params[0], params[3], "k--", label="Fit", color="r")

    # Add graph elements
    plt.xticks(np.arange(0, params[4] + 1))
    plt.xlabel(xlabel="Time [\u03BCs]")  # Unicode chr for mu
    plt.ylabel(ylabel="Counts")
    plt.title(label=f"Muon Lifetime Counts for Run {params[5]} with {params[2]}\u03BC Bins")
    plt.legend(loc="upper right")

    flpath = f"MuonData_Run{params[5]}_Bin{params[6]}"
    plt.savefig(flpath, bbox_inches='tight')
    plt.close()    # Prevents overlap of figures during save

    # plt.show()    # Debug

    return flpath


def data_analysis(pth, run, trim, bins=20):

    data = import_excel(pth)

    max_val = float(np.ceil(max(data)))
    step = max_val / bins

    y, bin_edges = np.histogram(data, bins=np.arange(0, max_val + step, step=step))

    if trim != 0:
        idx = int(trim/step)
        y = y[idx:]
        bin_edges = bin_edges[idx:]

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    sigma = np.sqrt(y)    # Uncertainty

    # Program crashes if bin has no values -> Sigma = 0 !DivideByZero
    for i in np.arange(len(sigma)):
        if sigma[i] == 0:
            print(f"Bin empty at index {i}... Adjusting sigma to prevent divide by zero error")
            sigma[i] = (sigma[i-1] + sigma[i+1]) / 2    # Chooses error close averaged between adjacent bins

    # Calculate best fit of data
    p0 = np.array([5000, 2.1, 150])    # Guess initial values for fit parameters
    p, cov = opt.curve_fit(theory_model, bin_centers, y, p0, sigma)    # Returns parameter values and Covariance Matrix
    expected = theory_model(bin_centers, *p)    # Plug new parameters into theory to get line of best fit

    # Chi Square Calc
    chisq = np.sum(((y - expected)/sigma)**2)    # Calculate chi-square value
    dof = len(y) - 3
    chi_red = chisq/dof

    # tbl_data(bin_centers, y, sigma, expected, run, step)   # Debug -- Generate console table of data
    # print(f"Chi^2 = {chisq:9.4f}")
    # print(f"Reduced Chi^2 = {chi_red:9.4f}\n")

    plot_input = [bin_centers, y, step, expected, max_val, run, bins]
    plt_path = plot_data(plot_input)
    store_data(len(data), p, cov, run, plt_path, chisq, chi_red)


def conclusions():
    with open("muon_data_analysis.json", 'r') as file:
        data = json.load(file)

    tau_vals = []
    for run in data:
        print(run)
        # if run != "bins":
        #     print(tau)
        #     chi = run["chisq"]
        #     tau = run["tau"]
        #     tau_unc = run["tau unc"]


def main(runs, bins, trim):
    make_json(runs, bins, trim)

    for i in range(len(runs)):
        run = i+1
        path = runs[i]
        for bin in bins:
            data_analysis(path, run, trim[i], bin)

    conclusions()


runs = [
    "/Users/zeke/PycharmProjects/MPL_Muons/MuonData_1-28-20_Fmtd.xlsx",         # Data for 1-28-20
    "/Users/zeke/PycharmProjects/MPL_Muons/MuonData_1-30_2-3-1_Fmtd.xlsx",      # Data for 1-30-20
    "/Users/zeke/PycharmProjects/MPL_Muons/MuonData_2-4-20_Fmtd.xlsx",          # Data for 2-4-20
    "/Users/zeke/PycharmProjects/MPL_Muons/MuonData_2-6-20_Fmtd.xlsx"           # Data for 2-6-20
]
bins = [10, 20, 40, 80]
trim = [0, 0, 1, 1]

main(runs, bins, trim)
