import os
import time
import torch
import numpy as np
import pickle
import gc
from tqdm import tqdm
from pathlib import Path
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import logging

# Timing helper function
def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def init_model(config_file, checkpoint_file, device="cuda:0"):
    """
    Wrapper function to initialize a model.
    Similar to init_model in switching_overhead_bak.py.
    """
    logger = common_utils.create_logger('model_init')
    logger.setLevel(logging.WARNING)
    
    cfg_from_yaml_file(config_file, cfg)
    cfg.TAG = Path(config_file).stem
    
    # Build dataloader to get dataset
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=logger, training=False
    )
    
    # Build and initialize model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=checkpoint_file, logger=logger)
    model.cuda()
    model.eval()
    
    return model

def inference_detector(model, data_dict):
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    return pred_dicts

# Configs and checkpoints for testing switching overhead
branches = [['cfgs/waymo_models/centerpoint_dyn_pillar024_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar024_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar028_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar028_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar032_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar032_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar036_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar036_4x.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel100.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel100.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel110.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel110.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel120.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel120.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel125.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel125.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel130.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel130.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel140.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel140.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar038.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar038.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar040.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar040.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar042.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar042.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar044.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar044.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar046.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar046.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar048.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar048.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar050.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar050.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar052.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar052.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar054.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar054.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar056.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar056.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar058.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar058.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar060.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar060.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar066.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar066.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar070.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar070.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel036.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel036.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel038.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel038.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel040.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel040.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel042.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel042.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel044.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel044.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel046.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel046.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel048.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel048.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel050.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel050.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel052.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel052.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel054.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel054.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel056.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel056.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel058.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel058.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel060.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel060.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel062.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel062.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel064.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel064.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel066.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel066.pth']]

# Set parameters
branch_num = len(branches)
RUNTIMES = 1

def switching_w_buffering(test_loader):
    """Test switching overhead with buffering - all models loaded into GPU memory at once."""
    # Create logger only once at the beginning
    logger = common_utils.create_logger('switching_w_buffering')
    logger.setLevel(logging.WARNING)
    
    # Use all branches
    branch_count = branch_num
    
    # The switching overhead results will be a branch_count*branch_count matrix
    switching_overhead_w = np.zeros((branch_count, branch_count, RUNTIMES))
    
    # Get a sample data dict for inference
    data_dict = next(iter(test_loader))
    
    # Dictionary to store models
    models = {}
    
    # Load all models using wrapper function
    print(f"Loading {branch_count} models into GPU memory...")
    
    # First, load all models
    for i in tqdm(range(branch_count)):
        model_name = f"model_{i}"
        try:
            # Use init_model wrapper function
            models[model_name] = init_model(branches[i][0], branches[i][1])
            
            # Preheating with 5 inferences
            for _ in range(5):
                inference_detector(models[model_name], data_dict)
            
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            models[model_name] = None
    print(f"All {branch_count} models loaded and preheated successfully")
    
    # Check if any models were successfully loaded
    if all(model is None for model in models.values()):
        print("No models were successfully loaded, aborting test.")
        return np.zeros((0, 0))
    
    # Profiling
    for k in range(RUNTIMES):
        for i in range(branch_count):
            model_i_name = f"model_{i}"
            # Skip if model_i failed to load
            if models[model_i_name] is None:
                print(f"Skipping {model_i_name} as it failed to load")
                continue
                
            for j in range(branch_count):
                model_j_name = f"model_{j}"
                # Skip if model_j failed to load
                if models[model_j_name] is None:
                    print(f"Skipping {model_j_name} as it failed to load")
                    switching_overhead_w[i][j][k] = -1  # Mark as invalid result
                    continue

                # Do 10 inferences on model_i to warm it up (like in switching_overhead_bak.py)
                print(f"Warming up {model_i_name} before switching to {model_j_name}...")
                for _ in range(10):
                    inference_detector(models[model_i_name], data_dict)
                
                # Switching happens here - do 20 inferences on model_j
                inference_lat = []
                
                # First inference after switching
                time_0 = time_sync()
                inference_detector(models[model_j_name], data_dict)
                time_1 = time_sync()
                inference_lat.append(1000 * (time_1 - time_0))
                
                # Remaining 19 inferences
                for _ in range(19):
                    time_2 = time_sync()
                    inference_detector(models[model_j_name], data_dict)
                    time_3 = time_sync()
                    inference_lat.append(1000 * (time_3 - time_2))
                
                print(f'Branches before and after switching are: ({i}, {j})')
                overhead = inference_lat[0] - np.mean(inference_lat[1:])
                switching_overhead_w[i][j][k] = overhead
                print(f"Switching overhead: {overhead:.2f} ms")
    
    # Compute mean overhead (excluding failed tests marked as -1)
    mask = switching_overhead_w >= 0
    switching_overhead_w_mean = np.zeros((branch_count, branch_count))
    for i in range(branch_count):
        for j in range(branch_count):
            valid_values = switching_overhead_w[i, j, mask[i, j]]
            if len(valid_values) > 0:
                switching_overhead_w_mean[i, j] = np.mean(valid_values)
            else:
                switching_overhead_w_mean[i, j] = -1  # No valid measurements
    
    return switching_overhead_w_mean


def main():
    # Create logger
    logger = common_utils.create_logger()
    logger.setLevel(logging.WARNING)
    
    # Load the first config to set up dataloader
    cfg_from_yaml_file(branches[0][0], cfg)
    cfg.TAG = Path(branches[0][0]).stem
    
    # Build dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4, logger=logger, training=False
    )
    
    # Create output directory
    output_dir = Path('output/switching_overhead')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test switching overhead with buffering
    print("Testing switching overhead with buffering...")
    switching_overhead_w_mean = switching_w_buffering(test_loader)
    if switching_overhead_w_mean.size > 0:  # Only save if we got results
        np.save(str(output_dir) + '/switching_overhead_w_buffer', switching_overhead_w_mean)
        print("Switching overhead with buffering testing completed!")
    else:
        print("Switching overhead with buffering test failed to produce results.")

if __name__ == '__main__':
    main()