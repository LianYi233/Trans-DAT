import os


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wyn/myStudy/2302-DAT/TransT-wyn/ltr'    # Base directory for saving network checkpoints.
        self.save_pth_dir = '/data_m1/tmp_data/2302-DAT/'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        # self.lasot_dir = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-Fused'    # 8-21
        self.use_hsi = True
        self.NIR_dir = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-NIRRedNIR-FalseColor'    # 12-09
        self.VIS_dir = '/home/wyn/myData/01-hyperspectral/challenge2023/datasets/training/HSI-VIS-FalseColor'

        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''      # 'path_of_COCO2017'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
