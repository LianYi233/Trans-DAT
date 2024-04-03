"""
my version 2023-11-22 of visualization all bboxes.
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
    data_type = 'VIS'   # NIR / RedNIR / VIS
    data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/HSI-'+data_type+'-FalseColor'
    model2test1 = '/data_m1/tmp_data/2302-DAT/results/validation/final/txt/HSIAtt+Baseline+GRL-0045-'+data_type+'/'
    model2test2 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/MHT'
    model2test3 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/SiamBAN'
    model2test4 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/SiamCAR'
    model2test5 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/SiamGAT'
    model2test6 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/STARTK'
    model2test7 = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/model_to_test/baseline/TransT'
    # create dataset and gt boxes
    dataset_NIR = DatasetFactory.create_dataset(name=args.dataset, dataset_root=data_root, load_img=False)

    for v_idx, video in enumerate(dataset_NIR):
        print(video.name)
        # video.img_names = [x.split('.')[0] + '.jpg' for x in video.img_names]
        pred_bboxes1 = np.loadtxt(os.path.join(model2test1, video.name+'.txt'))
        pred_bboxes2 = np.loadtxt(os.path.join(model2test2, data_type, video.name + '.txt'))
        pred_bboxes3 = np.loadtxt(os.path.join(model2test3, data_type, video.name + '.txt'))
        pred_bboxes4 = np.loadtxt(os.path.join(model2test4, data_type, video.name + '.txt'))
        pred_bboxes5 = np.loadtxt(os.path.join(model2test5, data_type, video.name + '.txt'))
        pred_bboxes6 = np.loadtxt(os.path.join(model2test6, data_type, video.name + '.txt'))
        pred_bboxes7 = np.loadtxt(os.path.join(model2test7, data_type, video.name + '.txt'))

        for idx, (img, gt_bbox) in enumerate(video):
            false_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if idx == 0:
                cv2.destroyAllWindows()
                # no need to save the init frame
            if args.vis and idx > 0:
                gt_bbox = list(map(int, video.gt_traj[idx]))
                pred_bbox1 = list(map(int, pred_bboxes1[idx]))      # transfer to
                pred_bbox2 = list(map(int, pred_bboxes2[idx]))
                pred_bbox3 = list(map(int, pred_bboxes3[idx]))
                pred_bbox4 = list(map(int, pred_bboxes4[idx]))
                pred_bbox5 = list(map(int, pred_bboxes5[idx]))
                pred_bbox6 = list(map(int, pred_bboxes6[idx]))
                pred_bbox7 = list(map(int, pred_bboxes7[idx]))


                cv2.rectangle(false_img, (pred_bbox2[0], pred_bbox2[1]), (pred_bbox2[0] + pred_bbox2[2], pred_bbox2[1] + pred_bbox2[3]), (0, 0, 255), 2)      # pred2: blue
                cv2.rectangle(false_img, (pred_bbox3[0], pred_bbox3[1]), (pred_bbox3[0] + pred_bbox3[2], pred_bbox3[1] + pred_bbox3[3]), (255, 0, 255), 2)  # pred3: magenta
                cv2.rectangle(false_img, (pred_bbox4[0], pred_bbox4[1]), (pred_bbox4[0] + pred_bbox4[2], pred_bbox4[1] + pred_bbox4[3]), (255, 128, 0), 2)  # pred4: orange
                cv2.rectangle(false_img, (pred_bbox5[0], pred_bbox5[1]), (pred_bbox5[0] + pred_bbox5[2], pred_bbox5[1] + pred_bbox5[3]), (0, 255, 255), 2)  # pred5: light blue
                cv2.rectangle(false_img, (pred_bbox6[0], pred_bbox6[1]), (pred_bbox6[0] + pred_bbox6[2], pred_bbox6[1] + pred_bbox6[3]), (0, 199, 140), 2)  # pred6: Turkish blue
                cv2.rectangle(false_img, (pred_bbox7[0], pred_bbox7[1]), (pred_bbox7[0] + pred_bbox7[2], pred_bbox7[1] + pred_bbox7[3]), (255, 255, 0), 2)  # pred7: yellow

                cv2.rectangle(false_img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 2)  # gt: green
                cv2.rectangle(false_img, (pred_bbox1[0], pred_bbox1[1]), (pred_bbox1[0] + pred_bbox1[2], pred_bbox1[1] + pred_bbox1[3]), (255, 0, 0), 2)      # pred1(Ours): red

                cv2.putText(false_img, '#' + str(idx + 1), (26, 47), cv2.FONT_HERSHEY_SIMPLEX, 2, (88, 87, 86), 2)
                cv2.putText(false_img, '#'+str(idx+1), (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)   # cv2.bitwise_not(false_img)[40][40]
                # cv2.imshow(video.name, cv2.cvtColor(false_img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(10000)
                save_path = os.path.join(model2test1, '../../Figs_new', data_type, video.name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/{:04d}.png'.format(idx+1), cv2.cvtColor(false_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # save results
        # result_path = os.path.join(save_path, '{}.txt'.format(video.name))
        # with open(result_path, 'w') as f:
        #     for x in pred_bboxes:
        #         f.write('\t'.join([str(i) for i in x]) + '\n')


if __name__ == '__main__':
    main()
