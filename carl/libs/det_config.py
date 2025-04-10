from pathlib import Path

import yaml
from easydict import EasyDict
import pickle
import os


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        if new_config['_BASE_CONFIG_'] == 'cfgs/dataset_configs/waymo_dataset.yaml':
            # new_config['_BASE_CONFIG_'] = '/anvil/projects/x-cis230283/adaptive-3d-openpcdet/tools/cfgs/dataset_configs/waymo_dataset.yaml'
            new_config['_BASE_CONFIG_'] = '/depot/schaterj/data/3d/work_dir/adaptive-3d-openpcdet-baseline/tools/cfgs/dataset_configs/waymo_dataset.yaml'      
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)
        
        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0


def check_sequence_name_with_all_version(sequence_file):
    if not sequence_file.exists():
        found_sequence_file = sequence_file
        for pre_text in ['training', 'validation', 'testing']:
            if not sequence_file.exists():
                temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                if temp_sequence_file.exists():
                    found_sequence_file = temp_sequence_file
                    break
        if not found_sequence_file.exists():
            found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
        if found_sequence_file.exists():
            sequence_file = found_sequence_file
    return sequence_file


def include_waymo_data(root_path, split, data_path, mode='test'):
    all_infos = []
    #self.logger.info('Loading Waymo dataset')
    #split_dir = root_path / 'ImageSets' / (split + '.txt')
    if mode == 'test':
        split_dir = os.path.join(root_path, 'ImageSets/val_val.txt')
    elif mode == 'test_test':
        split_dir = os.path.join(root_path, 'ImageSets/val_test.txt')
    else:
        raise NotImplementedError
    
    sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]    
    
    
    waymo_infos = []
    seq_name_to_infos = {}

    num_skipped_infos = 0
    for k in range(len(sample_sequence_list)):
        sequence_name = os.path.splitext(sample_sequence_list[k])[0]
        info_path = data_path / sequence_name / ('%s.pkl' % sequence_name)
        info_path = check_sequence_name_with_all_version(info_path)
        if not info_path.exists():
            num_skipped_infos += 1
            continue
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            waymo_infos.extend(infos)

        seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

    all_infos.extend(waymo_infos[:])
    return all_infos