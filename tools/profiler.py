import os
# For debug purpose, set as 1. Otherwise, use 0
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

config = 'cfgs/waymo_models/centerpoint_dyn_pillar020_1x.yaml'
# Read the config file
cfg_from_yaml_file(config, cfg)
cfg.TAG = Path(config).stem
cfg.EXP_GROUP_PATH = '/'.join(config.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
np.random.seed(1024)
dist_test = False
total_gpus = 1
batch_size = 1
workers = 1

# Create logger
logger = common_utils.create_logger()

# Build the dataloader
test_set, test_loader, sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=batch_size,
    dist=dist_test, workers=workers, logger=logger, training=False)
dataset = test_loader.dataset
class_names = dataset.class_names

# slo_list = (100, 150, 200, 250, 300, 500)
slo_list = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
out_dir = '../output/oracle'
# models = ['waymo_mse_adamw_64', 'waymo_mse_sgd_16', 'waymo_mse_sgd_64']
models = ['waymo_ql_v2_40branches_acc','waymo_ql_v2_40branches_loss_random_latency_adjusted_sigmoid', 'waymo_ql_v2_40branches_loss_random_latency_adjusted', 'waymo_ql_v2_40branches_loss_sigmoid', 'waymo_ql_v2_40branches_loss']
for slo in tqdm(slo_list[::-1]):
    # det_path = os.path.join(out_dir, f'waymo_ql_v2_40branches_loss_sigmoid/e9_thresh{slo:d}.pkl')
    det_path = os.path.join(out_dir, f'oracle/orin_slo{slo:d}_oracle.pkl')
    final_output_dir = '../output/waymo_results/oracle/eval'
    os.makedirs(final_output_dir, exist_ok=True)
    # Read the detection results
    print('================', det_path, '=====================')
    det_annos = pickle.load(open(det_path, 'rb'))
    ret_dict = {}
    result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )

    ret_dict.update(result_dict)
    print(ret_dict)