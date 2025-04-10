import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
cp_branches = [['cfgs/waymo_models/centerpoint_dyn_pillar024_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar024_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar028_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar028_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar032_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar032_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar036_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar036_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar040_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar040_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar044_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar044_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar048_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar048_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar052_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar052_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar056_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar056_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar060_4x.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel100.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel100.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel150.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel150.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel200.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel200.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel250.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel250.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel300.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel300.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel350.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel350.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel400.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel400.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel450.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel450.pth']]

dsvt_branches = [['cfgs/waymo_models/dsvt_sampled_pillar020.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar020.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar030.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar030.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar040.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar040.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar050.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar050.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar060.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar060.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar070.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar070.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar080.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar080.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar090.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar090.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar100.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar100.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar110.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar110.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar120.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar120.pth'],
                ['cfgs/waymo_models/dsvt_sampled_pillar130.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_pillar130.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel020.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel020.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel030.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel030.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel040.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel040.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel050.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel050.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel060.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel060.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel070.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel070.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel080.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel080.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel090.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel090.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel100.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel100.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel110.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel110.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel120.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel120.pth'],
                ['cfgs/waymo_models/dsvt_sampled_voxel130.yaml',
                    '../output/waymo_checkpoints/dsvt_sampled_voxel130.pth']]

branches = cp_branches + dsvt_branches

# Timing helper function
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def loss_profiling():
    # Collect Loss Values
    logger = common_utils.create_logger()
    profiling_results = []
    for i, (config, ckpt) in enumerate(cp_branches[:2]):
        print(i)
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
        output_dir = cfg.ROOT_DIR / 'output' / 'profiling' / 'val'
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
        loss_dict = []
        progress_bar = tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)
        for idx, data_dict in enumerate(test_loader):
            # if idx == 1:
            #     break
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                loss, tb_dict, disp_dict = model.collect_loss(data_dict)
            #print('tb_dict', tb_dict)
            loss_dict.append(tb_dict)
            del data_dict
            progress_bar.update()
        del test_loader 
        #del logger

        # Save the loss values
        loss_result_dir = str(output_dir) + '/' + str(cfg.TAG) + '_loss.pkl'
        with open(loss_result_dir, 'wb') as f:
            pickle.dump(loss_dict, f)

def main():
    time0 = time_sync()
    loss_profiling()
    time1 = time_sync()
    print('Total profiling time', (time1 - time0))

if __name__ == '__main__':
    main()