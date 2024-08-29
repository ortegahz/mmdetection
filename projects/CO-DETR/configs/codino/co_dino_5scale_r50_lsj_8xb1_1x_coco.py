_base_ = ['co_dino_5scale_r50_lsj_8xb2_1x_coco.py']

train_dataloader = dict(batch_size=1, num_workers=1)