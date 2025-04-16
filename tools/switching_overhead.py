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

# Timing helper function
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Configs and checkpoints for testing switching overhead
# Using a subset of branches from inference.py for testing
branches = [['cfgs/waymo_models/centerpoint_dyn_pillar024_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar024_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar028_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar028_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar032_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar032_4x.pth'],
            ['cfgs/waymo_models/centerpoint_dyn_pillar036_4x.yaml',
                '../output/waymo_checkpoints/centerpoint_dyn_pillar036_4x.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel100.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel100.pth']]

# Set parameters
branch_num = len(branches)
RUNTIMES = 3

def switching_wo_buffering(test_loader):
    logger = common_utils.create_logger()
    # The switching overhead results will be a branch_num*branch_num matrix
    switching_overhead_wo = np.zeros((branch_num, branch_num, RUNTIMES))
    
    # Profiling
    for k in range(RUNTIMES):
        for i in range(branch_num):
            # Load the start model_i
            print(f'Load Model {i}')
            cfg_from_yaml_file(branches[i][0], cfg)
            cfg.TAG = Path(branches[i][0]).stem
            model_i = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_loader.dataset)
            model_i.load_params_from_file(filename=branches[i][1], logger=logger)
            model_i.cuda()
            model_i.eval()
            
            # Preheating with 10 inferences
            for idx, data_dict in enumerate(test_loader):
                if idx == 10:
                    break
                load_data_to_gpu(data_dict)
                with torch.no_grad():
                    model_i.forward(data_dict)
            
            for j in range(branch_num):
                # Switching happens here
                inference_lat = []
                
                # Load model_j only if j != i
                if j != i:
                    # First inference includes model loading time
                    time_0 = time_sync()
                    cfg_from_yaml_file(branches[j][0], cfg)
                    cfg.TAG = Path(branches[j][0]).stem
                    model_j = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_loader.dataset)
                    model_j.load_params_from_file(filename=branches[j][1], logger=logger)
                    model_j.cuda()
                    model_j.eval()
                    
                    # First inference after loading
                    data_dict = next(iter(test_loader))
                    load_data_to_gpu(data_dict)
                    with torch.no_grad():
                        model_j.forward(data_dict)
                    time_1 = time_sync()
                    inference_lat.append(1000 * (time_1 - time_0))
                    
                    # Remaining 19 inferences
                    for _ in range(19):
                        data_dict = next(iter(test_loader))
                        load_data_to_gpu(data_dict)
                        time_2 = time_sync()
                        with torch.no_grad():
                            model_j.forward(data_dict)
                        time_3 = time_sync()
                        inference_lat.append(1000 * (time_3 - time_2))
                    
                    # Clean up model_j
                    del model_j
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # Use the same model, just measure inference times
                    # First inference
                    data_dict = next(iter(test_loader))
                    load_data_to_gpu(data_dict)
                    time_0 = time_sync()
                    with torch.no_grad():
                        model_i.forward(data_dict)
                    time_1 = time_sync()
                    inference_lat.append(1000 * (time_1 - time_0))
                    
                    # Remaining 19 inferences
                    for _ in range(19):
                        data_dict = next(iter(test_loader))
                        load_data_to_gpu(data_dict)
                        time_2 = time_sync()
                        with torch.no_grad():
                            model_i.forward(data_dict)
                        time_3 = time_sync()
                        inference_lat.append(1000 * (time_3 - time_2))
                
                print(f'Branches before and after switching are: ({i}, {j})')
                overhead = inference_lat[0] - np.mean(inference_lat[1:])
                switching_overhead_wo[i][j][k] = overhead
            
            # Clean up model_i
            del model_i
            torch.cuda.empty_cache()
            gc.collect()
            print(f'Delete Model {i}')
    
    # Compute mean overhead
    switching_overhead_wo_mean = np.mean(switching_overhead_wo, axis=2)
    return switching_overhead_wo_mean

def switching_w_buffering(test_loader):
    logger = common_utils.create_logger()
    # The switching overhead results will be a branch_num*branch_num matrix
    switching_overhead_w = np.zeros((branch_num, branch_num, RUNTIMES))
    
    # Buffer all branches
    models = []
    for i in range(branch_num):
        print(f'Load Model {i}')
        cfg_from_yaml_file(branches[i][0], cfg)
        cfg.TAG = Path(branches[i][0]).stem
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_loader.dataset)
        model.load_params_from_file(filename=branches[i][1], logger=logger)
        model.cuda()
        model.eval()
        models.append(model)
    
    # Preheating with 5 inferences for each model
    for model in models:
        for idx, data_dict in enumerate(test_loader):
            if idx == 5:
                break
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                model.forward(data_dict)
    
    # Profiling
    for k in range(RUNTIMES):
        for i in range(branch_num):
            for j in range(branch_num):
                # Do 10 inferences on model_i
                for idx, data_dict in enumerate(test_loader):
                    if idx == 10:
                        break
                    load_data_to_gpu(data_dict)
                    with torch.no_grad():
                        models[i].forward(data_dict)
                
                # Switching happens here
                # Do 20 inferences on model_j
                inference_lat = []
                
                # First inference after switching
                data_dict = next(iter(test_loader))
                load_data_to_gpu(data_dict)
                time_0 = time_sync()
                with torch.no_grad():
                    models[j].forward(data_dict)
                time_1 = time_sync()
                inference_lat.append(1000 * (time_1 - time_0))
                
                # Remaining 19 inferences
                for _ in range(19):
                    data_dict = next(iter(test_loader))
                    load_data_to_gpu(data_dict)
                    time_2 = time_sync()
                    with torch.no_grad():
                        models[j].forward(data_dict)
                    time_3 = time_sync()
                    inference_lat.append(1000 * (time_3 - time_2))
                
                print(f'Branches before and after switching are: ({i}, {j})')
                overhead = inference_lat[0] - np.mean(inference_lat[1:])
                switching_overhead_w[i][j][k] = overhead
    
    # Clean up all models
    for i, model in enumerate(models):
        del model
        print(f'Delete Model {i}')
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compute mean overhead
    switching_overhead_w_mean = np.mean(switching_overhead_w, axis=2)
    return switching_overhead_w_mean

def main():
    # Create output directory
    output_dir = Path('output/switching_overhead')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = common_utils.create_logger()
    
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
    print(f'Total number of samples: \t{len(test_set)}')
    
    # Test switching overhead without buffering
    print("Testing switching overhead without buffering...")
    switching_overhead_wo_mean = switching_wo_buffering(test_loader)
    np.save(str(output_dir) + '/switching_overhead_wo_buffer', switching_overhead_wo_mean)
    
    # Test switching overhead with buffering
    print("Testing switching overhead with buffering...")
    switching_overhead_w_mean = switching_w_buffering(test_loader)
    np.save(str(output_dir) + '/switching_overhead_w_buffer', switching_overhead_w_mean)
    
    print("Switching overhead testing completed!")

if __name__ == '__main__':
    main()