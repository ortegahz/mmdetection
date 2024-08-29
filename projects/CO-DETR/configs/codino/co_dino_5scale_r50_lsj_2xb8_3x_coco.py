_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_coco.py']

max_epochs = 36

# param_scheduler = [dict(milestones=[30])]
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs)

train_dataloader = dict(batch_size=4, num_workers=4)
