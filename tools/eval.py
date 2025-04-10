import numpy as np
import argparse
import pickle
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pathlib import Path
from tqdm import tqdm
import copy
import os
# Allow tensorflow GPU memory usage to allocate gradually
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
# os.environ['OMP_NUM_THREADS'] = '8'
# os.environ['MKL_NUM_THREADS'] = '8'

from pcdet.datasets.waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--predictions', type=str, default=None, help='pickle file with predictions')
    parser.add_argument('--config', type=str, default=None, help='yaml config file for the model')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--output', type=str, default=None, help='output txt file')
    parser.add_argument('--per_frame', type=str, default="False", help='per frame analysis')
    parser.add_argument('--div', type=int, default=1024, help='interval per restart (the algorithm speeds up calculation when restarted, but restarts require time)')
    parser.add_argument('--start', type=int, default=0, help='start frame of ground truth (0-indexed)')
    args = parser.parse_args()

    pd = pickle.load(open(args.predictions, 'rb'))
    cfg_from_yaml_file(args.config, cfg)
    cfg.TAG = Path(args.config).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.config.split('/')[1:-1])

    dataset = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.batch, dist=False, training=False)[1].dataset
    class_names = dataset.class_names
    gt = [copy.deepcopy(info['annos']) for info in dataset.infos]
    gt = gt[args.start:args.start+len(pd)]

    ev = OpenPCDetWaymoDetectionMetricsEstimator()
    rt_uni = dict()

    if args.per_frame == "True":
        split = [int(i) for i in np.linspace(0,len(pd),args.div)]
        file = open(args.output, "w")
        for i in tqdm(range(len(split)-1)):
            ev.waymo_evaluation_per_frame(pd[split[i]:split[i+1]], gt[split[i]:split[i+1]], class_names, 1000, fake_gt_infos=dataset.dataset_cfg.get('INFO_WITH_FAKELIDAR', False), start=split[i], file=file)
        file.close()
    else:
        rt_uni[-1] = ev.waymo_evaluation(pd, gt, class_names, 1000, fake_gt_infos=dataset.dataset_cfg.get('INFO_WITH_FAKELIDAR', False))
        rt_uni = '\n'.join([f'{i} : {str(v)}' for i,v in rt_uni.items()])
        open(args.output, "w").write(rt_uni)

if __name__ == '__main__':
    main()
