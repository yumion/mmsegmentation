default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
    save_dir=
    '/home/ishikawa/miccai_challenge_2023/Syn-ISS/mmsegmentation/configs/endovis2023_syn_iss_3colors'
)
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = './work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_syn_iss_3colors-1333x750/best_mIoU_epoch_17.pth'
resume = False
tta_model = None
num_classes = 4
crop_size = (
    540,
    960,
)
num_epochs = 30
warmup_ratio = 0.01
dataset_type = 'EndoVisSynISSDataset'
train_data_root = '/data1/shared/miccai/EndoVis2023/Syn-ISS/dataset-1/train/'
val_data_root = '/data1/shared/miccai/EndoVis2023/Syn-ISS/dataset-1/test/'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        132.1977451413987,
        72.76279601595651,
        42.36444859421867,
    ],
    std=[
        11.874730079673917,
        7.303477043371616,
        5.920501229709389,
    ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(
        540,
        960,
    ))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(
            960,
            540,
        ),
        ratio_range=(
            0.5,
            2.0,
        ),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(
        540,
        960,
    ), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        960,
        540,
    ), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='EndoVisSynISSDataset',
        data_root='/data1/shared/miccai/EndoVis2023/Syn-ISS/dataset-1/train/',
        data_prefix=dict(
            img_path='images', seg_map_path='3colors_masks_palette'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(
                    960,
                    540,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomCrop', crop_size=(
                    540,
                    960,
                ), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EndoVisSynISSDataset',
        data_root='/data1/shared/miccai/EndoVis2023/Syn-ISS/dataset-1/test/',
        data_prefix=dict(
            img_path='images', seg_map_path='3colors_masks_palette'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                960,
                540,
            ), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='EndoVisSynISSDataset',
        data_root='/data1/shared/miccai/EndoVis2023/Syn-ISS/dataset-1/test/',
        data_prefix=dict(
            img_path='images', seg_map_path='3colors_masks_palette'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                960,
                540,
            ), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ]))
val_evaluator = dict(
    type='IoUMetric', iou_metrics=[
        'mIoU',
    ])
test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
    ],
    output_dir=
    '/home/ishikawa/miccai_challenge_2023/Syn-ISS/mmsegmentation/configs/endovis2023_syn_iss_3colors/pred',
    keep_results=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=50))
auto_scale_lr = dict(enable=False, base_batch_size=16)
depths = [
    2,
    2,
    18,
    2,
]
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = dict({
    'backbone':
    dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'absolute_pos_embed':
    dict(lr_mult=0.1, decay_mult=0.0),
    'relative_position_bias_table':
    dict(lr_mult=0.1, decay_mult=0.0),
    'query_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'query_feat':
    dict(lr_mult=1.0, decay_mult=0.0),
    'level_embed':
    dict(lr_mult=1.0, decay_mult=0.0),
    'backbone.stages.0.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.2.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.3.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.4.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.5.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.6.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.7.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.8.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.9.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.10.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.11.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.12.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.13.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.14.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.15.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.16.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.blocks.17.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.0.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.3.blocks.1.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.0.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.1.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0),
    'backbone.stages.2.downsample.norm':
    dict(lr_mult=0.1, decay_mult=0.0)
})
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.005,
    eps=1e-08,
    betas=(
        0.9,
        0.999,
    ))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.005,
        eps=1e-08,
        betas=(
            0.9,
            0.999,
        )),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone':
            dict(lr_mult=0.1, decay_mult=1.0),
            'backbone.patch_embed.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed':
            dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table':
            dict(lr_mult=0.1, decay_mult=0.0),
            'query_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat':
            dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed':
            dict(lr_mult=1.0, decay_mult=0.0),
            'backbone.stages.0.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.2.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.3.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.4.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.5.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.6.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.7.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.8.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.9.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.10.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.11.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.12.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.13.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.14.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.15.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.16.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.blocks.17.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.0.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.3.blocks.1.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.0.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.1.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.stages.2.downsample.norm':
            dict(lr_mult=0.1, decay_mult=0.0)
        }),
        norm_decay_mult=0.0))
param_scheduler = [
    dict(type='LinearLR', start_factor=1, begin=0, end=1500, by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=30,
        T_max=30,
        eta_min_ratio=0.01,
        by_epoch=True,
        convert_to_iter_based=True),
]
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[
            132.1977451413987,
            72.76279601595651,
            42.36444859421867,
        ],
        std=[
            11.874730079673917,
            7.303477043371616,
            5.920501229709389,
        ],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(
            540,
            960,
        )),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        depths=[
            2,
            2,
            18,
            2,
        ],
        num_heads=[
            4,
            8,
            16,
            32,
        ],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        with_cp=False,
        frozen_stages=-1),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[
            128,
            256,
            512,
            1024,
        ],
        strides=[
            4,
            8,
            16,
            32,
        ],
        feat_channels=256,
        out_channels=256,
        num_classes=4,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=5.0,
                        pred_act=True,
                        eps=1.0),
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
launcher = 'pytorch'
work_dir = '/home/ishikawa/miccai_challenge_2023/Syn-ISS/mmsegmentation/configs/endovis2023_syn_iss_3colors'
