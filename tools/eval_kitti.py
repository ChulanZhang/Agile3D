from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pathlib import Path
import sys
import pickle
import copy
import numpy as np
from pcdet.utils import common_utils
from tqdm import tqdm


if len(sys.argv) < 5:
	print("Usage: python eval.py predictions.pkl config.yaml batch_size output.txt")
	exit()

with open(sys.argv[1], "rb") as f:
	pd = pickle.load(f)

cfg_from_yaml_file(sys.argv[2], cfg)
cfg.TAG = Path(sys.argv[2]).stem
cfg.EXP_GROUP_PATH = '/'.join(sys.argv[2].split('/')[1:-1])
logger = common_utils.create_logger()
test_set, test_loader, sampler = build_dataloader(
	dataset_cfg=cfg.DATA_CONFIG,
	class_names=cfg.CLASS_NAMES,
	batch_size=int(sys.argv[3]),
	dist=False, logger=logger, training=False
)

dataset = test_loader.dataset
class_names = dataset.class_names
gt = [copy.deepcopy(info['annos']) for info in dataset.infos]

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.datasets.kitti import kitti_utils
map_name_to_kitti = {'Vehicle': 'Car',
					'Pedestrian': 'Pedestrian',
					'Cyclist': 'Cyclist',
					'Sign': 'Sign',
					'Car': 'Car'}

kitti_utils.transform_annotations_to_kitti_format(pd, map_name_to_kitti=map_name_to_kitti)
kitti_utils.transform_annotations_to_kitti_format(
	gt, map_name_to_kitti=map_name_to_kitti,
	info_with_fakelidar=dataset.dataset_cfg.get('INFO_WITH_FAKELIDAR', False))
kitti_class_names = [map_name_to_kitti[x] for x in class_names]

# print(len(pd))
# print(len(gt))

acc = []
for i in tqdm(range(len(pd))):
	ap_result_str, ap_dict = kitti_eval.get_official_eval_result(gt_annos=[gt[i]], dt_annos=[pd[i]], current_classes=kitti_class_names)
	mAP = np.mean([ap_dict['Car_3d/easy_R40'], ap_dict['Car_3d/moderate_R40'], ap_dict['Car_3d/hard_R40'],
				   ap_dict['Pedestrian_3d/easy_R40'], ap_dict['Pedestrian_3d/moderate_R40'], ap_dict['Pedestrian_3d/hard_R40'],
				   ap_dict['Cyclist_3d/easy_R40'], ap_dict['Cyclist_3d/moderate_R40'], ap_dict['Cyclist_3d/hard_R40']])
	acc.append(mAP)


with open(sys.argv[4], 'w') as file:
    for item in acc:
        file.write(str(item) + '\n')

