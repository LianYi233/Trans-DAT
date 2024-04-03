""""
my version 2023 MAY.
one-pass evaluation.
S and P AUC curves of OPE.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset
from toolkit.evaluation import OPEBenchmark
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':

    dataset = ['NIR', 'RedNIR', 'VIS']
    sample_capa = [30, 11, 46]
    tracker_names = ['MHT', 'SiamBAN', 'SiamCAR', 'SiamGAT', 'STARTK', 'TransT', 'HSIAtt+Baseline+GRL-0045']
    tracker_show_names = ['MHT', 'SiamBAN', 'SiamCAR', 'SiamGAT', 'STARTK', 'TransT', 'Trans-DAT(Ours)']
    color_to_show = ['chocolate', 'lightcoral', 'gold', 'red', 'lightseagreen', 'slateblue', 'navy']
    csv_path = '/data_m1/tmp_data/2302-DAT/results/validation/csv/'
    save_fig_path = '/data_m1/tmp_data/2302-DAT/results/validation/AUC/'

    thresholds_s = np.arange(0, 1.05, 0.01)  # IoU threshold ts
    thresholds_p = np.arange(0, 51, 1)  # CLE threshold tp

    data_type = 0
    # 1. read in csv data and draw S curve
    NUM = 7  # num of trackers to eval
    y_s = np.ndarray(shape=(NUM, sample_capa[data_type], len(thresholds_s)))
    for i in range(NUM):
        print(csv_path + dataset[data_type] + str(i + 1) + '_s.csv')
        df2 = pd.read_csv(csv_path + '/S/' + dataset[data_type] + str(i + 1) + '_s.csv', sep=',', header=None)
        for index, row in df2.iterrows():
            aa = row[1:].tolist()
            y_s[i, index] = aa

    # draw S curve
    for index in range(sample_capa[data_type]):
        for i in range(NUM):
            plt.plot(thresholds_s, y_s[i, index, :],
                     label=tracker_show_names[i]+str('[%.3f]'%y_s[i, index, :].mean()), color=color_to_show[i], ms=2, linewidth=3)
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        plt.title("Success plot of OPE\n", {'weight': 'bold', 'size': 17})
        plt.xlabel("Overlap threshold", {'size': 15})
        plt.ylabel("Success rate", {'size': 15})
        plt.legend(prop={'size': 14}, loc=0)  # bbox_to_anchor=(0.05, 0.05), loc=3
        plt.grid()
        # plt.show()
        plt.savefig(save_fig_path + index + '_s.eps', format='eps')  # bbox_inches='tight'
        plt.close()


    # # 2. read csv and draw P curve
    # NUM = 7  # num of trackers to eval
    # y_p = np.ndarray(shape=(NUM, len(gt_name), len(thresholds_p)))
    # for i in range(NUM):
    #     print(csv_path + '/P/' + dataset + str(i + 1) + '_p.csv')
    #     df2 = pd.read_csv(csv_path + '/P/' + dataset + str(i + 1) + '_p.csv', sep=',', header=None)
    #     for index, row in df2.iterrows():
    #         # df2 = pd.read_csv(csv_path + '/P/' + dataset + str(i + 1) + '_p.csv', sep=',', usecols=[id + 1],
    #         #                   header=None)  # index_col=gt_name[0]
    #         aa = row[1:].tolist()
    #         y_p[i, index] = aa
    #
    # for index in range(len(gt_name)):
    #     for i in range(NUM):
    #         plt.plot(thresholds_p, y_p[i, index, :],
    #                  label=tracker_show_names[i]+str('[%.3f]'%y_p[i, index, :].mean()), color=color_to_show[i], ms=2, linewidth=3)
    #
    #     plt.xlim(0, 50)
    #     plt.ylim(0, 1)
    #     plt.title("Precision plot of OPE\n", {'weight': 'bold', 'size': 17})
    #     plt.xlabel("Center location error threshold", {'size': 15})
    #     plt.ylabel("Precision", {'size': 15})
    #     plt.legend(prop={'size': 14}, loc=1)  # bbox_to_anchor=(0.05, 0.05), loc=3
    #     plt.grid()
    #     # plt.show()
    #     plt.savefig(save_fig_path + gt_name[index] + '_p.eps', format='eps')  # bbox_inches='tight'
    #     plt.close()


