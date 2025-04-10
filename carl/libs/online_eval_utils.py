from pathlib import Path
import copy
import os

from libs.det_config import cfg, cfg_from_yaml_file, include_waymo_data
from libs.det_eval_python import waymo_evaluation


def init_online_eval(branches, opt, split='val'):
    ## this function prepare the annotation and the gt for the online mAP evaluation
    # config_name = '/anvil/projects/x-cis230283/adaptive-3d-openpcdet/tools/cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml'
    config_name = '/depot/schaterj/data/3d/work_dir/adaptive-3d-openpcdet-baseline/tools/cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml'
    cfg_from_yaml_file(config_name, cfg)
    cfg.TAG = Path(config_name).stem
    cfg.EXP_GROUP_PATH = '/'.join(config_name.split('/')[1:-1])
    
    # Create the dataset and the dataloader
    # mode = 'test'
    #root_path = '/anvil/projects/x-cis230283/datasets/waymo'
    root_path = '/depot/schaterj/data/3d/work_dir/adaptive-3d-openpcdet-baseline/data/waymo'
    # split = cfg.DATA_CONFIG.DATA_SPLIT[mode]
    #data_path = Path('/anvil/projects/x-cis230283/adaptive-3d-openpcdet/data/waymo/waymo_processed_data_v0_5_0')
    data_path = Path('/depot/schaterj/data/3d/work_dir/adaptive-3d-openpcdet-baseline/data/waymo/waymo_processed_data_v0_5_0')
    
    if split == 'val': # the validation split
        infos = include_waymo_data(root_path, split, data_path, mode='test')
    elif split == 'test': # the test split
        # print('in test split: include_waymo_data')
        infos = include_waymo_data(root_path, split, data_path, mode='test_test')
    else:
        raise NotImplementedError
    #dataset = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.batch, dist=False, training=False)[1].dataset
    
    # prepare all the gt
    class_names = cfg.CLASS_NAMES
    waymo_gt = [copy.deepcopy(info['annos']) for info in infos]
    #print('len of waymo_gt:', len(waymo_gt), waymo_gt[0])
    #gt = gt[args.start:args.start+len(pd)]

    # assert the branches is equal to the class in the training
    assert len(branches) == opt['model']['num_classes']
    
    # profile_root = '/anvil/projects/x-cis230283/datasets/waymo_new_profiling'
    profile_root = '/depot/schaterj/data/3d/waymo_results/waymo_new_profiling'    
    if split == 'val':
        root = os.path.join(profile_root, 'det/val/')
    elif split == 'test':
        # print('in test split: path')
        root = os.path.join(profile_root, 'det/test/')
        # root = '/anvil/projects/x-cis230283/datasets/waymo_new_profiling/det/test/'
    else:
        raise NotImplementedError
    
    all_detection_res = []
    import pickle
    for b in branches: 
        curr_filename = os.path.join(root, (b + '_det.pkl'))
        with open(curr_filename, 'rb') as f:
            x = pickle.load(f)
            all_detection_res.append(x)    

    return all_detection_res, waymo_gt, class_names