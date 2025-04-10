import os
import time
import torch
import numpy as np
import pickle
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import threading
from utils.contention import GPUContentionGenerator

# Configs and checkpoints for all branches
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
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel150.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel150.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel200.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel200.pth'],       
            ['cfgs/waymo_models/dsvt_sampled_pillar028.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar028.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar030.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar030.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar032.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar032.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar034.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar034.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar036.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar036.pth'],
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
            ['cfgs/waymo_models/dsvt_sampled_pillar076.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar076.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar080.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar080.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar086.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar086.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar090.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar090.pth'],
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
                '../output/waymo_checkpoints/dsvt_sampled_voxel066.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel068.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel068.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel070.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel070.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel080.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel080.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel090.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel090.pth']]

micro_branches =   [['cfgs/waymo_models/second_voxel100.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel100.pth'],
                    ['cfgs/waymo_models/second_voxel150.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel150.pth'],
                    ['cfgs/waymo_models/second_voxel200.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel200.pth'],
                    ['cfgs/waymo_models/second_voxel250.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel250.pth'],
                    ['cfgs/waymo_models/second_voxel300.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel300.pth'],
                    ['cfgs/waymo_models/second_voxel350.yaml',
                    '../output/waymo_checkpoints_micro/second_voxel350.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar032.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar032.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar036.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar036.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar040.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar040.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar044.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar044.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar048.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar048.pth'],
                    ['cfgs/waymo_models/pointpillar_4x_pillar052.yaml',
                    '../output/waymo_checkpoints_micro/pointpillar_4x_pillar052.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar032_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar032_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar036_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar036_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar040_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar040_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar044_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar044_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar048_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar048_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_hv_pillar052_4x.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_hv_pillar052_4x.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel100.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel100.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel150.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel150.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel200.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel200.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel250.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel250.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel300.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel300.pth'],
                    ['cfgs/waymo_models/centerpoint_without_resnet_hv_voxel350.yaml',
                    '../output/waymo_checkpoints_micro/centerpoint_without_resnet_hv_voxel350.pth']]

baselines = [['cfgs/waymo_models/centerpoint_pillar_1x.yaml',
                '../output/waymo_baselines/centerpoint_pillar_1x.pth'],
            ['cfgs/waymo_models/centerpoint_without_resnet.yaml',
                '../output/waymo_baselines/centerpoint_without_resnet.pth'],
            ['cfgs/waymo_models/second.yaml',
                '../output/waymo_baselines/second.pth'],
            ['cfgs/waymo_models/PartA2.yaml',
                '../output/waymo_baselines/PartA2.pth'],
            ['cfgs/waymo_models/pointpillar_1x.yaml',
                '../output/waymo_baselines/pointpillar_1x.pth'],
            ['cfgs/waymo_models/pv_rcnn.yaml',
                '../output/waymo_baselines/pv_rcnn.pth']]

nus_branches = [['cfgs/adapt3d_nus/cbgs_cp_pp020_dv.yaml',
                    '../output/nus_checkpoints/cbgs_cp_pp020_dv.pth'],
                ['cfgs/adapt3d_nus/cbgs_cp_pp030_dv.yaml',
                    '../output/nus_checkpoints/cbgs_cp_pp030_dv.pth'],
                ['cfgs/adapt3d_nus/cbgs_cp_pp040_dv.yaml',
                    '../output/nus_checkpoints/cbgs_cp_pp040_dv.pth'],
                ['cfgs/adapt3d_nus/cbgs_cp_pp050_dv.yaml',
                    '../output/nus_checkpoints/cbgs_cp_pp050_dv.pth'],
                ['cfgs/adapt3d_nus/cbgs_cp_pp060_dv.yaml',
                    '../output/nus_checkpoints/cbgs_cp_pp060_dv.pth'],
                ['cfgs/adapt3d_nus/dsvt_pillar030_nus.yaml',
                    '../output/nus_checkpoints/dsvt_pillar030_nus.pth'],
                ['cfgs/adapt3d_nus/dsvt_pillar040_nus.yaml',
                    '../output/nus_checkpoints/dsvt_pillar040_nus.pth'],
                ['cfgs/adapt3d_nus/dsvt_pillar050_nus.yaml',
                    '../output/nus_checkpoints/dsvt_pillar050_nus.pth'],
                ['cfgs/adapt3d_nus/dsvt_pillar060_nus.yaml',
                    '../output/nus_checkpoints/dsvt_pillar060_nus.pth'],
                ['cfgs/adapt3d_nus/dsvt_pillar070_nus.yaml',
                    '../output/nus_checkpoints/dsvt_pillar070_nus.pth']]

# Timing helper function
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def latency_profiling(gpu_util):
    logger = common_utils.create_logger()
    profiling_results = []
    for i, (config, ckpt) in tqdm(enumerate(branches[34:])):
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
        workers = 8

        # Create results directory
        # output_dir = cfg.ROOT_DIR / 'output' / 'profiling' / cfg.TAG
        output_dir = cfg.ROOT_DIR / 'output' / 'iclr2025'/ 'lat_profiling_w_contention'/ f'{gpu_util}'
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / 'log.txt'
        #logger = common_utils.create_logger()

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
        model.eval()
        
        # Preheating
        for idx, data_dict in enumerate(test_loader):
            if idx == 9:
                break
            load_data_to_gpu(data_dict)
            with torch.no_grad():
                pred_dicts, _ = model.forward(data_dict)

        # Inference
        class_names = test_set.class_names
        det_annos = []
        e2e_lat = []
        progress_bar = tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)
        for idx, data_dict in enumerate(test_loader):
            load_data_to_gpu(data_dict)
            start_time = time_sync()
            with torch.no_grad():
                pred_dicts, _ = model.forward(data_dict)
            inference_time = 1000 * (time_sync() - start_time)
            e2e_lat.append(inference_time)
            if idx == 1000 and np.mean(e2e_lat) > 600:
                with open(log_dir, 'a') as file:
                    file.write(f'Branch Timeout: {cfg.TAG} Latency {np.mean(e2e_lat)}\n')
                print(f'Branch Timeout: {cfg.TAG} Latency {np.mean(e2e_lat)}\n')
                break
            annos = test_set.generate_prediction_dicts(data_dict, pred_dicts, class_names, None)
            det_annos += annos
            del data_dict
            progress_bar.update()
        del test_loader 
        del model
        #del logger

        # Save results: 1. detection results; 2. latency results
        det_result_dir = str(output_dir) + '/' + str(cfg.TAG) + '_det.pkl'
        with open(det_result_dir, 'wb') as f:
            pickle.dump(det_annos, f)
        
        lat_result_dir = str(output_dir) + '/' + str(cfg.TAG) + '_lat.pkl'
        with open(lat_result_dir, 'wb') as f:
            pickle.dump(e2e_lat, f)

        e2e_lat_mean = np.mean(e2e_lat)
        profiling_results.append([cfg.TAG, e2e_lat_mean])
    np.save(str(output_dir) + '/latency_profiling_val', profiling_results)

def main():
    gpu_util_to_level = {
        #10: 4096,
        #20: 8888,
        #30: 15000,
        #40: 21000,
        #50: 26888,
        #60: 33888,
        70: 39888,
        #80: 45200,
        #90: 52200,
        #95: 55500,
        #98: 58000,
        #99: 66666
        }
     # Create an instance of the GPUContentionGenerator
    gpu_contention_gen = GPUContentionGenerator(initial_level=4096, cpu_cores=[11])

    time0 = time_sync()
    # Loop through the GPU utilization levels and update the workload level
    for util, level in gpu_util_to_level.items():
        print(f"Setting GPU utilization to {util}% with level {level}")
        gpu_contention_gen.level = level  # Update the level
        # Start the GPU contention
        contention_thread = threading.Thread(target=gpu_contention_gen.start_contention)
        contention_thread.start()
        
        # Profiling
        latency_profiling(util)
        #time.sleep(30)

        # Free GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stop the GPU contention
        gpu_contention_gen.stop_contention()
        # Wait for the contention thread to finish
        contention_thread.join()

    time1 = time_sync()
    print('Total profiling time', (time1 - time0))

if __name__ == '__main__':
    main()
