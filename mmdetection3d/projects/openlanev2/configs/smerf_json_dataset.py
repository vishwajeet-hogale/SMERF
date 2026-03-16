# ==============================================================================
# Configuration for OpenLaneJSONDataset (Direct JSON Loading)
# This config uses the OpenLaneJSONDataset instead of Collection-based loading
# ==============================================================================

custom_imports = dict(imports=['projects.openlanev2.baseline'])

# Point cloud range and voxel size
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]
voxel_size = [0.2, 0.2, 8]

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

class_names = ['centerline']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7

# Lane encoding parameters
method_para = dict(n_points=11)
code_size = 3 * method_para['n_points']

# Model dimensions
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
)

_num_levels_ = 4
_num_heads_ = 4
bev_h_ = 100
bev_w_ = 200

# Model configuration
model = dict(
    type='BaselineMapGraph',
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    map_encoder=dict(
        type='MapGraphTransformer',
        input_dim=360,
        dmodel=_dim_,
        hidden_dim=_dim_,
        nheads=_num_heads_,
        nlayers=6,
        batch_first=True,
        pos_encoder=dict(
            type='SineContinuousPositionalEncoding',
            num_feats=16,
            temperature=1000,
            normalize=True,
            range=[point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]],
            offset=[point_cloud_range[0], point_cloud_range[1]],
        ),
    ),
    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_,
        num_cams=num_cams,
        embed_dims=_dim_,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        rotate_center=[bev_h_//2, bev_w_//2],
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_)),
                    dict(
                        type='MaskedCrossAttention',
                        embed_dims=_dim_,
                        num_heads=_num_heads_,),
                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn_graph', 'norm', 'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),
    bbox_head=dict(
        type='TEDeformableDETRHead',
        num_query=100,
        num_classes=13,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=_dim_),
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_)
                    ],
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.5),
        loss_iou=dict(type='GIoULoss', loss_weight=1.0),
        test_cfg=dict(max_per_img=50)),
    pts_bbox_head=dict(
        type='LCDeformableDETRHead',
        num_classes=1,
        in_channels=_dim_,
        num_query=100,
        bev_h=bev_h_,
        bev_w=bev_w_,
        sync_cls_avg_factor=False,
        with_box_refine=False,
        with_shared_param=False,
        code_size=code_size,
        code_weights=[1.0 for i in range(code_size)],
        pc_range=point_cloud_range,
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='LaneDetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0075),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    lclc_head=dict(
        type='RelationshipHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
    lcte_head=dict(
        type='RelationshipHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
)

# Data pipelines
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ResizeFrontView'),
    dict(type='CustomPadMultiViewImage', size_divisor=32),
    dict(type='CustomParametrizeSDMapGraph', method='even_points_onehot_type', method_para=dict(n_points=11)),
    dict(type='CustomDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'map_graph', 'onehot_category'],
        meta_keys=[
            'scene_token', 'sample_idx', 'img_paths',
            'img_shape', 'scale_factor', 'pad_shape',
            'lidar2img', 'can_bus',
        ],
    )
]

# Dataset configuration with OpenLaneJSONDataset
dataset_type = 'OpenLaneJSONDataset'
data_root = 'D:/TopoNet/data/OpenLane-V2'  # Path to split folders (train/, val/, test/)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split='val',  # Use 'val' or 'test' split
        lazy_load=True,  # Use lazy loading to reduce startup memory
        pipeline=test_pipeline,
        test_mode=True,
        modality=input_modality,
        decoding_function=dict(
            type='bezier_prediction_decode',
            method_para=method_para,
        ),
    ),
)

# Evaluation configuration
evaluation = dict(
    interval=1,
    save_best='OpenLane-V2 Score',
    rule='greater',
)

# Runner configuration
runner = dict(type='EpochBasedRunner', max_epochs=1)

log_level = 'INFO'
