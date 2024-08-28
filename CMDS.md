# infer

python demo/image_demo.py demo/demo.jpg \
    configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    --weights /home/Huangzhe/test/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --device cpu

python demo/image_demo.py \
    demo/demo.jpg \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
    --weights /home/Huangzhe/test/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
    --device cuda:0

python demo/image_demo.py \
    demo/demo.jpg \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
    --weights /home/Huangzhe/test/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
    --device cuda:0

python demo/image_demo.py \
    demo/demo.jpg \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py \
    --weights /home/Huangzhe/test/co_dino_5scale_r50_lsj_8xb2_1x_coco/epoch_12.pth \
    --device cuda:0

python demo/video_demo.py demo/demo.mp4 \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py \
    /home/Huangzhe/test/co_dino_5scale_r50_lsj_8xb2_1x_coco/epoch_12.pth \
    --out /home/Huangzhe/test/result.mp4

python demo/video_demo.py /home/Huangzhe/test/Z-D-30m-002.mp4 \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_fire.py \
    /home/Huangzhe/test/work_dirs/co_dino_5scale_r50_lsj_8xb2_1x_fire/epoch_32.pth \
    --out /home/Huangzhe/test/result.mp4

python demo/image_demo.py /root/autodl-tmp/BOSH-FM数据采集-samples-merge \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_2xb1_16e_o365tococo2fire.py \
    --weights /root/autodl-tmp/work_dirs/co_dino_5scale_swin_l_2xb1_16e_o365tococo2fire/epoch_15.pth \
    --device cuda:1

python demo/video_demo.py /root/autodl-tmp/X-170m-001.mp4 \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_2xb1_16e_o365tococo2fire.py \
    /root/autodl-tmp/work_dirs/co_dino_5scale_swin_l_2xb1_16e_o365tococo2fire/epoch_15.pth \
    --out /root/autodl-tmp/X-170m-001-res.mp4 \
    --device cuda:0

# wandb

4ca6289368165f23d11b994b443ed6e1af5ad70b
https://wandb.ai/

# test

bash tools/dist_test.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
    /home/Huangzhe/test/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
    8

bash tools/dist_test.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py \
    /home/Huangzhe/test/co_dino_5scale_r50_lsj_8xb2_1x_coco/epoch_12.pth \
    8

# train

bash ./tools/dist_train.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py \
    2

bash ./tools/dist_train.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py \
    8

bash ./tools/dist_train.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_voc.py \
    8

bash ./tools/dist_train.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_fire.py \
    8

screen bash ./tools/dist_train.sh \
    projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_2xb1_o365tococo2fire.py \
    2

screen bash ./tools/dist_train.sh \
    configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py \
    2

# data

python tools/misc/download_dataset.py --dataset-name coco2017 --save-dir /dev/shm/data

python tools/dataset_converters/pascal_voc.py /home/Huangzhe/test/voc_fire/VOCdevkit -o /home/Huangzhe/test/voc_fire/VOCdevkit/annotations --out-format coco
