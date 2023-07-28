_base_ = [
    "../_base_/default_runtime.py",
]

load_from = "/data2/src/atsushi/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_sararrp50-1333x750_lr1e-4/epoch_30.pth"  # noqa

num_classes = 4
crop_size = (750, 1333)  # height, width
num_epochs = 30
warmup_ratio = 0.01

dataset_type = "EndoVisDataset"
data_root = "/data2/shared/miccai"


# dataset settings
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[0.32858519781689005 * 255, 0.15265839395622285 * 255, 0.14655234887549404 * 255],
    std=[0.07691241763785549 * 255, 0.053818967599625046 * 255, 0.056615884572508365 * 255],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(1920, 1333), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root + "/EndoVis2017/train",
                data_prefix=dict(img_path="left_frames", seg_map_path="3colors"),
                dump_path="train_files_2017.csv",
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root + "/EndoVis2018/train",
                data_prefix=dict(img_path="left_frames", seg_map_path="3colors"),
                dump_path="train_files_2018.csv",
                pipeline=train_pipeline,
            ),
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root + "/EndoVis2017/test",
                data_prefix=dict(img_path="left_frames", seg_map_path="3colors"),
                dump_path="test_files_2017.csv",
                pipeline=test_pipeline,
            ),
            dict(
                type=dataset_type,
                data_root=data_root + "/EndoVis2018/test",
                data_prefix=dict(img_path="left_frames", seg_map_path="3colors"),
                dump_path="test_files_2018.csv",
                pipeline=test_pipeline,
            ),
        ],
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

# training schedule by epochs
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=num_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=True, interval=1, max_keep_ckpts=1, save_best="mIoU"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(
        type="SegVisualizationHook",
        draw=True,
        interval=100,
    ),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# runtime config
tta_model = None
log_processor = dict(by_epoch=True)

depths = [2, 2, 18, 2]
# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    "backbone": dict(lr_mult=0.1, decay_mult=1.0),
    "backbone.patch_embed.norm": backbone_norm_multi,
    "backbone.norm": backbone_norm_multi,
    "absolute_pos_embed": backbone_embed_multi,
    "relative_position_bias_table": backbone_embed_multi,
    "query_embed": embed_multi,
    "query_feat": embed_multi,
    "level_embed": embed_multi,
}
custom_keys.update(
    {
        f"backbone.stages.{stage_id}.blocks.{block_id}.norm": backbone_norm_multi
        for stage_id, num_blocks in enumerate(depths)
        for block_id in range(num_blocks)
    }
)
custom_keys.update(
    {f"backbone.stages.{stage_id}.downsample.norm": backbone_norm_multi for stage_id in range(len(depths) - 1)}
)
# optimizer
optimizer = dict(type="AdamW", lr=0.00001, weight_decay=0.005, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0),
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1,
        begin=0,
        end=1500,
        by_epoch=False,
    ),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        end=num_epochs,
        T_max=num_epochs,
        eta_min_ratio=1e-2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[128, 256, 512, 1024],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(num_feats=128, normalize=True),  # SinePositionalEncoding
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),  # SinePositionalEncoding
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, dropout_layer=None, batch_first=True
                ),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)