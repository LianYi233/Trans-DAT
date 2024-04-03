"""
2024-03-14
check frame duration, spatial resolution etc of the dataset.
"""
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import os

import cv2
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
# from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', type=str, default='LaSOT', help='datasets')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', default='True', action='store_true', help='whether visualize result')
parser.add_argument('--name', default='transt', type=str, help='name of results')
args = parser.parse_args()

# torch.set_num_threads(1)
USE_HSI = True      # and remember to set self.use_hsi = True in toolkit/dataset/video.py

def main():
    # load config
    TorV = 'validation'  # training or validation
    # data_types = ['NIR', 'RedNIR', 'VIS']        # 'NIR', 'RedNIR',

    for data_type in ['NIR', 'RedNIR', 'VIS']:
        data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/' + TorV + '/HSI-' + data_type + '-FalseColor'
        # create dataset
        dataset_NIR = DatasetFactory.create_dataset(name=args.dataset, dataset_root=data_root, load_img=False)

        # load dataset
        for img_nir_dataset in dataset_NIR:
            img_nir_dataset.use_hsi = USE_HSI
            # img_nir_dataset.img_names = [x.split('.')[0]+'.jpg' for x in img_nir_dataset.img_names]
            for (img, gt_bbox_nir) in img_nir_dataset:
                img_nir_dataset.use_hsi = USE_HSI
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_nir))
                gt_bbox_nir = [cx - w / 2, cy - h / 2, w, h]
                init_info = {
                    'init_bbox': gt_bbox_nir,
                }
                break
            break

        frame_duration = []
        max_by_w = [0, 0]
        max_by_h = [0, 0]
        min_by_w = [9999, 9999]
        min_by_h = [9999, 9999]
        img_shape_min = [9999, 9999]
        img_shape_max = [1, 1]
        for v_idx, video in enumerate(dataset_NIR):
            # print(video)
            # frames_of_current_video = 0
            for idx, (img, gt_bbox) in enumerate(video):
                if img.shape[0] * img.shape[1] > img_shape_max[0] * img_shape_max[1]:
                    img_shape_max = img.shape[0:2]
                    # print(img_shape_max)
                if img.shape[0] * img.shape[1] < img_shape_min[0] * img_shape_min[1] and img.shape[1] * img.shape[0] > 0:
                    img_shape_min = img.shape[0:2]
                # if gt_bbox[2] > max_by_w[0]:
                #     max_by_w = gt_bbox[2:]
                #     # print(idx, gt_bbox[2:])
                # if gt_bbox[3] > max_by_h[1]:
                #     max_by_h = gt_bbox[2:]
                # if gt_bbox[2] < min_by_w[0] and gt_bbox[2] > 0:
                #     min_by_w = gt_bbox[2:]
                #     # print(idx, gt_bbox[2:])
                # if gt_bbox[3] < min_by_h[1] and gt_bbox[3] > 0:
                #     min_by_h = gt_bbox[2:]
                # frames_of_current_video += 1
            # frame_duration.append(frames_of_current_video)
        # print(data_type, min_by_w, min_by_h, max_by_w, max_by_h)
        print(data_type, img_shape_min, img_shape_max)


if __name__ == '__main__':
    main()
    # a = [778, 101, 182, 150, 155, 102, 93, 160, 185, 82, 127, 159, 151, 68, 138, 93, 119, 132, 164, 111, 105, 122, 178, 155, 271, 142, 184, 165, 124, 160]
    # b = [250, 375, 275, 450, 325, 250, 250, 250, 275, 400, 400]
    # c = [250, 625, 186, 471, 601, 131, 326, 976, 101, 131, 331, 930, 375, 275, 149, 731, 450, 725, 325, 501, 279, 1111, 530, 363, 552, 184, 117, 278, 250, 306, 363, 901, 800, 250, 250, 275, 400, 336, 210, 526, 396, 376, 601, 221, 400, 1209]
    # print(np.min(a), np.max(a), np.min(b), np.max(b), np.min(c), np.max(c))
