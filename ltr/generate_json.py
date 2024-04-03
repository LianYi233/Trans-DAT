"""
Create LaSOT.json for my dataset.
modify: training / validation, and NIR / RedNIR / VIS in data_root
"""

import json
import os

import numpy as np

data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/HSI-NIRRedNIR-FalseColor/'
data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/HSI-NIR-FalseColor/'
data_root = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/validation/HSI-RedNIR-FalseColor/'

class_list = os.listdir(data_root)
class_list.sort()

data_dict = {}
for class_item in class_list:
    if class_item.endswith('json') or class_item.endswith('zip'):
        continue
    class_path = os.path.join(data_root, class_item)
    img_list = os.listdir(class_path)
    img_list.sort()
    class_dict = {
        'video_dir': class_item,
        'init_rect': [],
        'img_names': [],
        'gt_rect': [],
        'attr': [],
        'absent': []
    }
    gt_file = os.path.join(class_path, 'groundtruth_rect.txt')
    if not os.path.exists(gt_file):
        gt_file = os.path.join(class_path, 'init_rect.txt')
    gt = np.loadtxt(gt_file, dtype=np.int32)
    descript_file = os.path.join(class_path, 'description.txt')
    if not os.path.exists(descript_file):
        attr_list = []
    else:
        attr_list = []
        with open(descript_file, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')
                attr_list.extend([x for x in line if x != ''])
        class_dict['attr'] = attr_list

    img_name_list, absent_list = [], []
    for index in range(len(img_list)-1, -1, -1):
        if img_list[index].endswith('txt') or img_list[index].endswith('db'):
            del img_list[index]
    for index in range(1, len(img_list)+1):
        img_index = '{:04d}.png'.format(index) if 'FalseColor' not in data_root else '{:04d}.jpg'.format(index)
        assert img_index in img_list
        img_name = os.path.join(class_item, img_index)
        img_name_list.append(img_name)
        absent_list.append(1)
    if len(gt.shape) == 1:
        gt = gt.reshape(1, -1)
    class_dict['gt_rect'] = [[int(x) for x in y] for y in gt]
    class_dict['absent'] = list(absent_list)
    class_dict['img_names'] = img_name_list
    class_dict['init_rect'] = [int(x) for x in gt[0]]
    data_dict[class_item] = class_dict

with open(os.path.join(data_root, 'LaSOT.json'), 'w', encoding='utf-8') as f:
    json.dump(data_dict, f)
print(os.path.join(data_root, 'LaSOT.json'))
