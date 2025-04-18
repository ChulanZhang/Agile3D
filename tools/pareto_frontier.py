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

# Configs and checkpoints for all branches
agile3d_branches = [['cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel100.yaml',
                '../output/waymo_checkpoints/centerpoint_without_resnet_dyn_voxel100.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar066.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar066.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar048.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar048.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel058.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel058.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel048.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel048.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel040.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel040.pth'],
            ['cfgs/waymo_models/dsvt_sampled_voxel038.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel038.pth']]

baselines = [['cfgs/waymo_models/dsvt_sampled_voxel032.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_voxel032.pth'],
            ['cfgs/waymo_models/dsvt_sampled_pillar032.yaml',
                '../output/waymo_checkpoints/dsvt_sampled_pillar032.pth'],
            ['cfgs/waymo_models/centerpoint_pillar_1x.yaml',
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


# Timing helper function
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def latency_profiling(branches):
    logger = common_utils.create_logger()
    profiling_results = []
    for i, (config, ckpt) in tqdm(enumerate(agile3d_branches + baselines)):
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
        output_dir = cfg.ROOT_DIR / 'output' / 'exp3'
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

    time0 = time_sync()
   
    # Profiling
    latency_profiling([agile3d_branches + baselines])

    time1 = time_sync()
    print('Total profiling time', (time1 - time0))

if __name__ == '__main__':
    main()
