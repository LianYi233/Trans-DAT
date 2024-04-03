"""
my version 2023.11 for OPE
modify: args.tracker_path; tracker_prefix(name); dataset_root (NIR/Red/V)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys

# env_path = os.path.join(os.path.dirname(__file__), '..')
# if env_path not in sys.path:
#     sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot_toolkit.toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from pysot_toolkit.toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot_toolkit.toolkit.visualization import draw_success_precision
import numpy as np

TorV = 'validation'     # training or validation
parser = argparse.ArgumentParser(description='transt evaluation')
parser.add_argument('--tracker_path', '-p', type=str, default='/home/wyn/myData/01-hyperspectral/challenge2023/datasets/'+TorV+'/model_to_test/',      # path_to_pred (remember to add .pth here)
                        help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, default='LaSOT',
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='nirrednir-smlr-3transt-0007-NIRRedNIR',   # 3transtpth / MHT / SiamBAN / SiamCAR / SiamGAT / STARTK / TransT
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.set_defaults(show_video_level=True)
args = parser.parse_args()


def main():
    tracker_dir = args.tracker_path
    trackers = glob(os.path.join(args.tracker_path, args.tracker_prefix))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, dataset_root='/home/wyn/myData/01-hyperspectral/challenge2023/datasets/'+TorV+'/HSI-VIS-FalseColor')   # training / validation / NIR / RedNIR / VIS
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            ret = benchmark.eval_success(trackers)
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision, trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision, trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            draw_success_precision(success_ret,
                                   name=dataset.name,
                                   videos=dataset.attr['ALL'],
                                   attr='ALL',
                                   precision_ret=precision_ret,
                                   norm_precision_ret=norm_precision_ret)

        precision = precision_ret[args.tracker_prefix]
        key_message = ''
        value_message = ''
        for key in precision:
            key_message += '{}\t'.format(key)
            value_message += '{:.3f},'.format(np.mean(precision[key]))
        print('{}: {}'.format(key_message, value_message))
        with open(r'/data_m1/tmp_data/2302-DAT/results/record.csv', 'a+') as f:
            f.write(value_message + '\n')
    else:
        exit(0)


if __name__ == '__main__':
    main()
