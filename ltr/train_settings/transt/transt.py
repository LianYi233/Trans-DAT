import numpy as np
import torch
# from ltr.dataset import Lasot, MSCOCOSeq, Got10k, TrackingNet
from ltr.dataset import NIR, VIS
from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.transt as transt_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
import os
import torch.nn as nn


def run(settings):
    # Most common settings are assigned in the settings struct
    if settings.env.use_hsi:
        mean_dict = [157.36511324823607, 125.40444488620267, 82.2771533998519,
                     0.25214408509098407, 0.1843704454171751, 0.14015638289476298]
        std_dict = [70.36119187090083, 65.16828134429738, 53.3996147107771,
                    0.781632971305069, 0.5896179391655295, 0.4535531479918772]
    else:
        mean_dict = [157.36511324823607, 125.40444488620267, 82.2771533998519]        # fc only
        std_dict = [70.36119187090083, 65.16828134429738, 53.3996147107771]
    settings.device = 'cuda'     # 'cuda'
    settings.description = 'TransT with default settings.'
    settings.batch_size = 24        # default: 8
    settings.num_workers = 20
    settings.multi_gpu = True
    settings.print_interval = 1
    settings.search_area_factor = 4.0
    settings.template_area_factor = 2.0
    settings.search_feature_sz = 32
    settings.template_feature_sz = 16
    settings.search_sz = settings.search_feature_sz * 8
    settings.temp_sz = settings.template_feature_sz * 8
    settings.center_jitter_factor = {'search': 3, 'template': 0}
    settings.scale_jitter_factor = {'search': 0.25, 'template': 0}

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    # Train datasets
    NIR_train = NIR(settings.env.NIR_dir, split='train')
    VIS_train = VIS(settings.env.VIS_dir, split='train')

    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05, use_hsi=settings.env.use_hsi))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2, normalize=False),
                                    tfm.Normalize(mean=mean_dict, std=std_dict))

    # Data processing to do on the training pairs
    data_processing_train = processing.TransTProcessing(search_area_factor=settings.search_area_factor,
                                                        template_area_factor = settings.template_area_factor,
                                                        search_sz=settings.search_sz,
                                                        temp_sz=settings.temp_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    # The sampler for training
    dataset_train = sampler.TransTSampler([VIS_train, NIR_train], [1, 1],  # train with fc # 10-08
                                          samples_per_epoch=1000 * settings.batch_size, max_gap=100,
                                          processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # Create network and actor
    model = transt_models.transt_resnet50(settings)

    # # Calculate FLOPs and Params.
    # from thop import profile
    # input_template = torch.randn(1, 6, 128, 128)
    # input_search = torch.randn(1, 6, 256, 256)
    # input_dt = torch.randn(1)
    # input_dt = np.int64(input_dt)
    # input_dt = torch.from_numpy(input_dt)
    # flops, params = profile(model, (input_template, input_search, input_dt, 0))
    # print("FLOPs: %.2fG" % (flops / 1e9))
    # print("Params: %.2fM" % (params / 1e6))

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.transt_loss(settings)

    actor = actors.TranstActor(net=model, objective=objective)
    # custom_ops = {nn.MultiheadAttention: get_complexity_MHA}

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-6,
        },
    ]

    # todo: set the backbone and domain learning rate.
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-6},
        {"params": [p for n, p in model.named_parameters() if 'domain' in n], 'lr': 1e-5},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4,     # 1e-5 ->
                                  weight_decay=1e-4)        # 1e-4
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)      # step_size = 1/2 max_epoch

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train], optimizer, settings, lr_scheduler, log_file=settings.log_file)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(200, load_latest=True, fail_safe=True)
