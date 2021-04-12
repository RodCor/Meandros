import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.patches as mpatches


class Statistics:
    def __init__(self, sample1, sample2, bin_width, test_choice):
        self.p_d_1 = sample1['PD'].values
        self.f_fmax_1 = sample1['f_f_max'].values
        self.p_d_2 = sample2['PD'].values
        self.f_fmax_2 = sample2['f_f_max'].values
        remove = 0
        if self.p_d_1[0] < self.p_d_2[0]:
            start = int(self.p_d_2[0] - 1)
            for i in self.p_d_1:
                if i < start:
                    remove += 1
            self.p_d_1 = self.p_d_1[remove:].tolist()
            self.f_fmax_1 = self.f_fmax_1[remove:].tolist()
            self.p_d_2 = self.p_d_2.tolist()
            self.f_fmax_2 = self.f_fmax_2.tolist()
        else:
            start = int(self.p_d_1[0] - 1)
            for i in self.p_d_2:
                if i < start:
                    remove += 1
            self.p_d_2 = self.p_d_2[remove:].tolist()
            self.f_fmax_2 = self.f_fmax_2[remove:].tolist()
            self.p_d_1 = self.p_d_1.tolist()
            self.f_fmax_1 = self.f_fmax_1.tolist()

        delta = bin_width
        bins = range(start, 101, delta)

        bins1 = pd.cut(self.p_d_1, bins=list(bins))
        bins2 = pd.cut(self.p_d_2, bins=list(bins))

        indexes1 = []
        index1 = 0
        for i in (bins1.value_counts()):
            index1 += i
            indexes1.append(index1)
        print(indexes1)

        indexes2 = []
        index2 = 0
        for i in (bins2.value_counts()):
            index2 += i
            indexes2.append(index2)
        print(indexes2)

        muestras11 = [self.f_fmax_1[i: j] for i, j in zip([0] +
                                                          indexes1, indexes1 + [None])]
        muestras21 = [self.f_fmax_2[i: j] for i, j in zip([0] +
                                                          indexes2, indexes2 + [None])]
        p_value1 = []

        if test_choice == "Mann-Whitney":
            for i in range(0, len(muestras11)):
                try:
                    u_statistic, pVal = stats.mannwhitneyu(muestras21[i], muestras11[i])
                    # ValueError==True
                    p_value1.append(-np.log10(pVal))
                    print(pVal, 'try')
                except:
                    p_value1.append(1)
                    print(0, 'except')
        elif test_choice == "Student":
            for i in range(0, len(muestras11)):
                try:
                    u_statistic, pVal = stats.t(muestras21[i], muestras11[i])
                    # ValueError==True
                    p_value1.append(-np.log10(pVal))
                    print(pVal, 'try')
                except:
                    p_value1.append(1)
                    print(0, 'except')
        
        v_elbow = 19.591
        v_wrist = 38.725
        elbow_err = 3.89
        wrist_err = 3.39

        clrs1 = ['red' if (x < 1.35) else 'green' for x in p_value1]  # < o > fijarse

        plt.style.use('ggplot')

        patchg1 = mpatches.Patch(color='green', label=r'$\mathcal{f}\ / \mathcal{f}_{max}$ (%)')
        patchr1 = mpatches.Patch(color='red')
        patch2 = mpatches.Patch(color='green', label=r'$\mathcal{f}\ / \mathcal{w}$ (%)')
        patch3 = mpatches.Patch(color='green', label=r'$\mathcal{f/w}\ / \mathcal{f/w}_{max}$ (%) ')

        fig1, axes1 = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        fig1.suptitle(test_choice, fontsize=16)

        axes1.bar(list(range(start, 101, delta)), p_value1, width=0.8 * delta,
                  color=clrs1)
        axes1.set_xlabel('PD position')
        axes1.set_ylabel(r'$\mathcal{-log(p)}$', fontsize=15, labelpad=30)
        axes1.tick_params(axis='both', which='major', labelsize=14)
        axes1.set_yscale('log')

        axes1.axvline(0, ls='--', color='gray')
        axes1.axvline(0, ls='--', color='gray')
        axes1.axhline(1.3, ls='--', color='gray')
        axes1.axvline(v_elbow, ls=':', color='gray')
        axes1.axvline(v_wrist, ls=':', color='gray')
        axes1.axvspan(v_elbow - elbow_err, v_elbow + elbow_err, alpha=0.5, color='#7B7D7D')
        axes1.axvspan(v_wrist - wrist_err, v_wrist + wrist_err, alpha=0.5, color='#B3B6B7')

        plt.show()

