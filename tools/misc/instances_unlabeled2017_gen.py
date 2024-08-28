import os

from mmengine.fileio import load, dump

dir_root = '/dev/shm/data/coco/annotations'
anns_train = load(os.path.join(dir_root, 'instances_train2017.json'))
anns_unlabeled = load(os.path.join(dir_root, 'image_info_unlabeled2017.json'))
anns_unlabeled['categories'] = anns_train['categories']
output_file_path = os.path.join(dir_root, 'instances_unlabeled2017.json')
dump(anns_unlabeled, output_file_path)
