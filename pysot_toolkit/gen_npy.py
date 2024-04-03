"""
generate feature npy for 3 dataset.
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

torch.set_num_threads(1)


def main():
    # load config
    TorV = 'validation'  # training or validation
    data_types = ['VIS', 'NIR', 'RedNIR']
    epoch_num = '0001'
    # labels = []

    isPrim = True
    for data_type_i in range(len(data_types)):
        data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/'+TorV+'/HSI-'+data_types[data_type_i]+'-FalseColor'
        net_path = '/data_m1/tmp_data/2302-DAT/checkpoints/ltr/transt/Trans-DAT/TransT_ep'+epoch_num+'.pth.tar'

        # create model
        net = NetWithBackbone(net_path=net_path, use_gpu=True)
        tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

        # create dataset
        dataset_NIR = DatasetFactory.create_dataset(name=args.dataset,
                                                    dataset_root=data_root,
                                                    load_img=False)

        for img_nir_dataset in dataset_NIR:
            # img_nir_dataset.use_hsi = True
            # img_nir_dataset.img_names = [x.split('.')[0]+'.jpg' for x in img_nir_dataset.img_names]
            for (img, gt_bbox_nir) in img_nir_dataset:

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox_nir))
                gt_bbox_nir = [cx - w / 2, cy - h / 2, w, h]
                init_info = {
                    'init_bbox': gt_bbox_nir,
                }
                break
            break

        # OPE tracking
        dataset = dataset_NIR

        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            print(video.name)
            # video.img_names = [x.split('.')[0] + '.jpg' for x in video.img_names]
            for idx, (img, gt_bbox) in enumerate(video):
                # false_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # tic = cv2.getTickCount()
                if idx == 0:        # use 1st frame only
                    # labels.append(int(bool(data_type_i)))
                    out, zf = tracker.initialize(img, init_info, use_hsi=False)
                    zf = zf[0].tensors.cpu()        # NestedTensor2ndarray
                    z_crop_ = zf.detach().numpy()
                    # z_crop_ = np.ravel(z_crop_[0].cpu().numpy())
                    if isPrim:
                        z_crop_ = np.ravel(z_crop_)
                        z_crop = np.expand_dims(z_crop_, 0)
                        # print(z_crop.shape, '1')
                        isPrim = False
                    else:
                        new_z_crop = np.ravel(z_crop_)
                        # print(z_crop.shape, new_z_crop.shape)
                        new_z_crop = np.expand_dims(new_z_crop, 0)
                        z_crop = np.vstack((z_crop, new_z_crop))

    # labels = np.array(labels)
    np.save('/data_m1/tmp_data/Trans-DAT_'+str(epoch_num)+'.npy', z_crop)


if __name__ == '__main__':
    main()
