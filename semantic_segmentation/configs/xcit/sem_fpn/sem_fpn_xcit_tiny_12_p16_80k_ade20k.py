# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
_base_ = [
    '../../_base_/models/sem_fpn_xcit_p16.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11]
    ),
    neck=dict(
        in_channels=[192, 192, 192, 192],
        out_channels=192,
    ),
    decode_head=dict(
        in_channels=[192, 192, 192, 192],
        num_classes=150,
        feature_strides=[4, 8, 16, 32],
),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
