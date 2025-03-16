import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats


def mean_sd_plots(group_1, e, e_sem, w, w_sem):
    """
    Plot mean and standard deviation of two groups
    Args:
        group_1: pd.DataFrame Group 1 to plot
        e: int Value of the event
        e_sem: int Standard error of the event
        w: int Value of the window
        w_sem: int Standard error of the window
    Output:
        Plot of the mean and standard deviation of the two groups
    """
    mean_1, *_ = stats.binned_statistic(
        group_1["PD"].values,
        group_1["f_f_max"].values,
        statistic="mean",
        bins=group_1["PD"].values + 100,
    )

    std_1, *_ = stats.binned_statistic(
        group_1["PD"].values,
        group_1["f_f_max"].values,
        statistic="std",
        bins=group_1["PD"].values + 100,
    )
    bin_width_1 = _[0][1] - _[0][0]
    bin_centers_1 = _[0][1:] - bin_width_1 / 2

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(bin_centers_1, mean_1, label="Group 1", color="#145A32")
    ax1.fill_between(
        bin_centers_1, mean_1 - std_1, mean_1 + std_1, facecolor="#7FB3D5", alpha=0.5
    )
    ax1.set_ylabel(r"$\mathcal{f}\ / \mathcal{f}_{max}$ (%)", fontsize=20, labelpad=30)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
    ax1.axvline(0, ls="--", color="gray")
    ax1.axvline(e, ls=":", color="gray")
    ax1.axvline(w, ls=":", color="gray")
    ax1.axvspan(e - e_sem, e + e_sem, alpha=0.5, color="#7B7D7D")
    ax1.axvspan(w - w_sem, w + w_sem, alpha=0.5, color="#B3B6B7")
