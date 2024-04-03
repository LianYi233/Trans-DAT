"""
my version 2023-11-22 of evaluation fc.
modify: data_type (NIR / RedNIR / VIS) and save_path.
latest
"""
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import sys
import time

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
USE_HSI = True      # IMPORTANT! Remember to set self.use_hsi = True in toolkit/dataset/video.py


def show_heatmap(feature):
    heatmap = feature.sum(0) / feature.shape[0]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = 1.0 - heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # cv2.imshow('heatmap', heatmap)
    # cv2.waitKey(10000)

    return heatmap


def main():
    # load config
    TorV = 'validation'  # training or validation
    # data_types = ['NIR', 'RedNIR', 'VIS']        # 'NIR', 'RedNIR',
    for data_type in ['VIS']:    # , 'RedNIR', 'VIS']
        for epoch_num in [45]:      # [5, 15, 20, 25, 30, 35, 40, 45, 50]
            epoch_num = '{:04d}'.format(epoch_num) if epoch_num > 0 else '0001'
            data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/' + TorV + '/HSI-' + data_type + '-FalseColor'
            net_path = '/data_m1/tmp_data/2302-DAT/checkpoints/ltr/Trans-DAT/TransT_ep' + epoch_num + '.pth.tar'
            model_to_test = '/data_m1/tmp_data/2302-DAT/results/' + TorV + '/final/txt/HSIAttGRLTf-' + epoch_num

            # create model
            net = NetWithBackbone(net_path=net_path, use_gpu=True)

            tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

            # create dataset
            dataset_NIR = DatasetFactory.create_dataset(name=args.dataset,
                                                        dataset_root=data_root,
                                                        load_img=False)

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

            # OPE tracking
            dataset = dataset_NIR
            bbox_name = 'init_bbox'

            for v_idx, video in enumerate(dataset):
                # if args.video != '':
                #     # test one special video
                #     if video.name != args.video:
                #         continue
                # video.img_names = [x.split('.')[0] + '.jpg' for x in video.img_names]
                toc = 0
                pred_bboxes = []
                scores = []
                track_times = []
                for idx, (img, gt_bbox) in enumerate(video):
                    # false_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tic = cv2.getTickCount()
                    if idx == 0:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
                        # init_info = {'init_bbox': gt_bbox_}
                        init_info[bbox_name] = gt_bbox_
                        # tracker.initialize(img, init_info, use_hsi=USE_HSI)  # init model and load weight
                        out, zf = tracker.initialize(img, init_info, video_name=video.name, use_hsi=True)  # init model and load weight
                        # zf = zf[0].tensors.cpu().detach().numpy()
                        # cv2.imwrite('/data_m1/tmp_data/hm1/' + video.name + '_att' + str(idx) + '.jpg', show_heatmap(zf[0, 0:3, :, :]))
                        pred_bbox = gt_bbox_
                        scores.append(None)
                        pred_bboxes.append(pred_bbox)
                    else:
                        outputs = tracker.track(img)
                        pred_bbox = outputs['target_bbox']
                        pred_bboxes.append(pred_bbox)
                        scores.append(outputs['best_score'])

                    toc += cv2.getTickCount() - tic
                    track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                toc /= cv2.getTickFrequency()

                # save results
                # save_path = os.path.join(model_to_test + '-' + data_type)
                # if not os.path.isdir(save_path):
                #     os.makedirs(save_path)
                # result_path = os.path.join(save_path, '{}.txt'.format(video.name))
                # with open(result_path, 'w') as f:
                #     for x in pred_bboxes:
                #         f.write('\t'.join([str(i) for i in x]) + '\n')
                print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                    v_idx + 1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
