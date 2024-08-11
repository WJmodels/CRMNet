# pretrained = '/mnt/workspace/project/MMLAB/mmsegmentation/my_mmseg_pretrain_model/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
pretrained = None
# load_from = '/mnt/workspace/project/MMLAB/work_dirs/hpc052155_mask2former_swinv2-l/iter_40000.b'
# load_from = '/home/sunhnayu/lln/project/MMLAB/work_dirs/convnetv2/hpc05190921_mask2former_convnetv2-l/best_mIoU84.37_iter_18550.pth'
# load_from = '/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_pretrain_model/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth'
# load_from = '/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_pretrain_model/convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth'
load_from = '/home/sunhnayu/lln/project/MMLAB/work_dirs/convnetv2/test_hpc05281644_ClinicDB_freunfre_mask2former_RFA_up_convnetv2-l/top_mIoU_96.1400_iter_2700.pth'
# load_from = None

# dataset_type = 'MyADE20KDataset'
# num_classes=150
# data_root = '/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ADEChallengeData2016'
# test_dataloader_data_prefix=dict(img_path='images/validation',seg_map_path='annotations/validation')
# train_dataloader_data_prefix=dict(img_path='images/training', seg_map_path='annotations/training')
# val_dataloader_data_prefix=dict(img_path='images/validation',seg_map_path='annotations/validation')
# class_weight=[ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, ]
# val_batch_size=1
# reduce_zero_label=True
# # crop_size = (640,640,)
# crop_size = (512,512,)

dataset_type = 'MyDatasetPolyp'
num_classes=2
data_root = '/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/CVC_ClinicDB'
test_dataloader_data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test')
train_dataloader_data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train')
# val_dataloader_data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val')
val_dataloader_data_prefix=test_dataloader_data_prefix

class_weight=[1.0,1.0,0.1,]
val_batch_size=8
reduce_zero_label=False
crop_size = (384,384,)

data_preprocessor_size=crop_size
max_iters = 20000
val_interval = 50
save_best=['mIoU']
max_keep_ckpts=3
save_top_k=1
logger_interval = 10
checkpoint_interval=1000
train_batch_size=8


backbone_embed_multi = dict(decay_mult=0.0, lr_mult=0.1)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.1)



custom_keys = dict({
    'absolute_pos_embed':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone':
    dict(decay_mult=1.0, lr_mult=0.1),
    'backbone.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.patch_embed.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.0.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.1.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.10.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.11.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.12.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.13.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.14.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.15.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.16.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.17.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.2.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.3.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.4.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.5.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.6.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.7.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.8.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.blocks.9.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.2.downsample.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.0.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'backbone.stages.3.blocks.1.norm':
    dict(decay_mult=0.0, lr_mult=0.1),
    'level_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_embed':
    dict(decay_mult=0.0, lr_mult=1.0),
    'query_feat':
    dict(decay_mult=0.0, lr_mult=1.0),
    'relative_position_bias_table':
    dict(decay_mult=0.0, lr_mult=0.1)
})
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=data_preprocessor_size,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')

default_hooks = dict(
    # checkpoint=dict(
    #     by_epoch=False, interval=5000, save_best='mIoU',
    #     type='CheckpointHook'),
    checkpoint=dict(by_epoch=False, interval=checkpoint_interval, type='CheckpointHook',save_best=save_best,max_keep_ckpts=max_keep_ckpts,save_top_k=save_top_k),
    logger=dict(interval=logger_interval, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
depths = [
    2,
    2,
    18,
    2,
]
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
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
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    # backbone=dict(
    #     arch='large',
    #     frozen_stages=3,
    #     out_indices=(
    #         0,
    #         1,
    #         2,
    #         3,
    #     ),
    #     drop_path_rate=0.2,
    #     img_size=384,
    #     pretrained_window_sizes=[
    #         12,
    #         12,
    #         12,
    #         6,
    #     ],
    #     type='mmpretrain.SwinTransformerV2',
    #     window_size=[
    #         24,
    #         24,
    #         24,
    #         12,
    #     ]),
    backbone=dict(
        arch='large',
        drop_path_rate=0.0,
        # drop_path_rate=0.15,
        layer_scale_init_value=0.0,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        # frozen_stages=3,
        gap_before_final_norm=False, # sparseconvnetv2没有这个参数
        type='mmpretrain.ConvNeXt',
        use_grn=True),
    # neck=dict(
    #     type='MyESAMNeck',
    #     in_channels=[
    #         192,
    #         384,
    #         768,
    #         1536,
    #     ],
    # ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        # size=(
        #     640,
        #     640,
        # ),
        # size=(
        #     384,
        #     384,
        # ),
        size=data_preprocessor_size,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=True,
        enforce_decoder_input_project=True,
        feat_channels=256,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        loss_cls=dict(
            # class_weight=[
            #     1.0,
            #     1.0,
            #     0.1,
            # ],
            class_weight=class_weight,
            loss_weight=2.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            # loss_weight=5.0,  # TODO
            loss_weight=10.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            # type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                init_cfg=None,
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        im2col_step=64,
                        init_cfg=None,
                        norm_cfg=None,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=6),
            init_cfg=None,
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.MyMSDeformAttnPixelDecoder'), # TODO
        
        positional_encoding=dict(normalize=True, num_feats=128),
        strides=[
            4,
            8,
            16,
            32,
        ],
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        use_sigmoid=True,
                        weight=5.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=5.0),
                ],
                type='mmdet.HungarianAssigner'),
            importance_sample_ratio=0.75,
            num_points=12544,
            oversample_ratio=3.0,
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.0,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.0)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')

# 原来的学习率策略
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'absolute_pos_embed':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone':
            dict(decay_mult=1.0, lr_mult=0.1),
            'backbone.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.patch_embed.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.0.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.1.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.10.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.11.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.12.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.13.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.14.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.15.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.16.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.17.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.2.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.3.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.4.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.5.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.6.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.7.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.8.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.blocks.9.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.2.downsample.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.0.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'backbone.stages.3.blocks.1.norm':
            dict(decay_mult=0.0, lr_mult=0.1),
            'level_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_embed':
            dict(decay_mult=0.0, lr_mult=1.0),
            'query_feat':
            dict(decay_mult=0.0, lr_mult=1.0),
            'relative_position_bias_table':
            dict(decay_mult=0.0, lr_mult=0.1)
        }),
        norm_decay_mult=0.0),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.0001,
    type='AdamW',
    weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=160000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]



# 我的学习率策略1：使用预热阶段和修改后的多项式衰减调度器
# param_scheduler = [
#     # 线性学习率预热调度器
#     dict(type='LinearLR',
#          start_factor=0.001,
#          by_epoch=False,  # 按迭代更新学习率
#          begin=0,
#          end=500),  # 预热前 500 次迭代
#     # 主学习率调度器
#     dict(
#         begin=0.001,
#         by_epoch=False,
#         end=10000,
#         eta_min=1e-7,
#         power=0.9,
#         type='PolyLR'),
# ]

# 我的学习率策略2：余弦周期调度器
# param_scheduler = dict(
#     T_max=2000,
#     begin=0.001,
#     by_epoch=False,
#     eta_min=1e-07 ,
#     type='CosineAnnealingLR')

# optim_wrapper = dict(
#     clip_grad=None,
#     optimizer = dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
#     paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
#     type='OptimWrapper'
# )




randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=test_dataloader_data_prefix,
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2560,
                640,
            ), type='Resize'),
            dict(reduce_zero_label=reduce_zero_label, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2560,
        640,
    ), type='Resize'),
    dict(reduce_zero_label=reduce_zero_label, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=max_iters, type='IterBasedTrainLoop', val_interval=val_interval)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=reduce_zero_label, type='LoadAnnotations'),
    dict(
        max_size=2560,
        resize_type='ResizeShortestEdge',
        scales=[
            320,
            384,
            448,
            512,
            576,
            640,
            704,
            768,
            832,
            896,
            960,
            1024,
            1088,
            1152,
            1216,
            1280,
        ],
        type='RandomChoiceResize'),
    # TODO 子豪的这个字典不一样
    # dict(
    #     keep_ratio=True,
    #     ratio_range=(
    #         0.5,
    #         2.0,
    #     ),
    #     scale=(
    #         2048,
    #         1024,
    #     ),
    #     type='RandomResize'),
    # dict(type='RandomMosaic',prob=0.5),
    dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(
        # data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_prefix=train_dataloader_data_prefix,
        data_root=data_root,
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
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
        # data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_prefix=val_dataloader_data_prefix,
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2560,
                640,
            ), type='Resize'),
            dict(reduce_zero_label=reduce_zero_label, type='LoadAnnotations'),
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
work_dir = '../work_dirs/convnetv2/test_hpc05281644_ClinicDB_freunfre_mask2former_RFA_up_convnetv2-l'
