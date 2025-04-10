import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

# Configs and checkpoints for all branches

dsvt_branches = ['cfgs/waymo_models/dsvt_sampled_voxel060.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel060.pth']

# Timing helper function
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def bev_feature_collector():
    # Collect Loss Values
    logger = common_utils.create_logger()
    profiling_results = []
    
    config, ckpt = dsvt_branches[0], dsvt_branches[1]
    
    # Read the config file
    cfg_from_yaml_file(config, cfg)
    cfg.TAG = Path(config).stem
    cfg.EXP_GROUP_PATH = '/'.join(config.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist_test = False
    total_gpus = 1
    batch_size = 1
    workers = 4

    # Create results directory
    output_dir = cfg.ROOT_DIR / 'output' / 'profiling' / 'bev_feature'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test, workers=workers, logger=logger, training=False)
    print(f'Total number of samples: \t{len(test_set)}')

    # Build the model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    print(ckpt)
    model.load_params_from_file(filename=ckpt, logger=logger)
    model.cuda()
    # Use the train mode as we need the loss values
    model.train() 

    # Inference
    class_names = test_set.class_names
    progress_bar = tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)
    for idx, data_dict in enumerate(test_loader):
        if idx == 5:
            break
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            bev_feature = model.collect_bev_feature(data_dict)
        file_name = f"{idx:05d}.pt"
        file_path = f"../output/bev_feature/test/{file_name}" 
        torch.save(bev_feature, file_path, _use_new_zipfile_serialization=False)
        progress_bar.update()


def main():
    time0 = time_sync()
    bev_feature_collector()
    time1 = time_sync()
    print('Total profiling time', (time1 - time0))

if __name__ == '__main__':
    main()