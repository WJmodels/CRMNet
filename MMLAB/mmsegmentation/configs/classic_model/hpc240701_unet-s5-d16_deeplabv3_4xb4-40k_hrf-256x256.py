pretrained = None
load_from = '/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_pretrain_model/Classic_model/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_256x256_40k_hrf_20211210_202032-59daf7a4.pth'
# load_from = None

# crop_size = (1024,1024,)
crop_size = (384,384,)
pixel_decoder_type = "mmdet.RDMSDeformAttnPixelDecoder"
# pixel_decoder_type = "mmdet.MyMSDeformAttnPixelDecoder"
num_classes = 3
# num_classes = 2
max_iters = 20000
PolyLR_end = max_iters
# val_interval = 10
val_interval = 50
save_best=['mIoU']
save_top_k=3
max_keep_ckpts=1
checkpoint_interval=1000
logger_interval = 10
train_batch_size=8
val_batch_size=1
# save_best_class=1
reduce_zero_label = False

data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')


data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Ulcer'
dataset_type = 'MyDatasetUlcer'

# data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Cerebral_Infarction'
# dataset_type = 'MyDatasetCerebralInfarction'

# data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ISIC_2018_Task1'
# dataset_type = "MyDatasetISIC2018Task1"

test_data_prefix = dict(img_path='img_dir/test', seg_map_path='ann_dir/test')
train_data_prefix = dict(img_path='img_dir/train', seg_map_path='ann_dir/train')
val_data_prefix = dict(img_path='img_dir/val', seg_map_path='ann_dir/val')


default_hooks = dict(
    # checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    # logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    checkpoint=dict(by_epoch=False, interval=checkpoint_interval, type='CheckpointHook',save_best=save_best,max_keep_ckpts=max_keep_ckpts,save_top_k=save_top_k),
    # logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    logger=dict(interval=logger_interval, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    2336,
    3504,
)
launcher = 'none'
# load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=128,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        base_channels=64,
        conv_cfg=None,
        dec_dilations=(
            1,
            1,
            1,
            1,
        ),
        dec_num_convs=(
            2,
            2,
            2,
            2,
        ),
        downsamples=(
            True,
            True,
            True,
            True,
        ),
        enc_dilations=(
            1,
            1,
            1,
            1,
            1,
        ),
        enc_num_convs=(
            2,
            2,
            2,
            2,
            2,
        ),
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=5,
        strides=(
            1,
            1,
            1,
            1,
            1,
        ),
        type='UNet',
        upsample_cfg=dict(type='InterpConv'),
        with_cp=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=16,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=64,
        in_index=4,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        type='ASPPHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        170,
        170,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=PolyLR_end,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        # data_prefix=dict(
        #     img_path='images/validation',
        #     seg_map_path='annotations/validation'),
        # data_root='data/HRF',
        data_prefix=test_data_prefix,
        # data_root='data/ade/ADEChallengeData2016',
        data_root=data_root, 
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2336,
                3504,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(iou_metrics=['mIoU','mDice','mFscore',], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2336,
        3504,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=max_iters, type='IterBasedTrainLoop', val_interval=val_interval)
train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        dataset=dict(
            # data_prefix=dict(
            #     img_path='images/training',
            #     seg_map_path='annotations/training'),
            # data_root='data/HRF',
            data_prefix=train_data_prefix,
            # data_root='data/ade/ADEChallengeData2016',
            data_root=data_root,    
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    scale=(
                        512,
                        512,
                    ),
                    type='RandomResize'),
                dict(
                    cat_max_ratio=0.75,
                    crop_size=(
                        256,
                        256,
                    ),
                    type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs'),
            ],
            type=dataset_type),
        times=40000,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2336,
            3504,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=val_batch_size,
    dataset=dict(
        # data_prefix=dict(
        #     img_path='images/validation',
        #     seg_map_path='annotations/validation'),
        # data_root='data/HRF',
        data_prefix=val_data_prefix,
        # data_root='data/ade/ADEChallengeData2016',
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2336,
                3504,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = test_evaluator
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
# work_dir = '../work_dirs/classic_model/ISIC/hpc240611_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256'
work_dir = '../work_dirs/classic_model/Ulcer/hpc240701_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256'