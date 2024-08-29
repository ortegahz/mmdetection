_base_ = ['co_dino_semi_coco_detection.py']

detector = _base_.model

model = dict(
    _delete_=True,
    type='SemiBaseDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=1.0,
        cls_pseudo_thr=0.9,
        min_pseudo_bbox_wh=(1e-2, 1e-2)),
    semi_test_cfg=dict(predict_on='teacher'))

# 10% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'semi_anns/instances_train2017.1@10.json'
unlabeled_dataset.ann_file = 'semi_anns/' \
                             'instances_train2017.1@10-unlabeled.json'
unlabeled_dataset.data_prefix = dict(img='train2017/')
train_dataloader = dict(
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop', max_iters=180000, val_interval=5000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='MeanTeacherHook')]
