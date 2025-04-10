import numpy as np
import argparse
import pickle
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pathlib import Path
import copy
import pandas as pds
from numba import jit

@jit(nopython=True)
def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def generate_waymo_type_results(infos, class_names, is_gt=False, fake_gt_infos=True):
    def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
        """
        Args:
            boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

        Returns:
            boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        """
        w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
        boxes3d_lidar[:, 2] += h[:, 0] / 2
        return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

    frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
    for frame_index, info in enumerate(infos):
        if is_gt:
            box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
            if 'num_points_in_gt' in info:
                zero_difficulty_mask = info['difficulty'] == 0
                info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
                info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
                nonzero_mask = info['num_points_in_gt'] > 0
                box_mask = box_mask & nonzero_mask
            else:
                print('Please provide the num_points_in_gt for evaluating on Waymo Dataset '
                        '(If you create Waymo Infos before 20201126, please re-create the validation infos '
                        'with version 1.2 Waymo dataset to get this attribute). SSS of OpenPCDet')
                raise NotImplementedError

            num_boxes = box_mask.sum()
            box_name = info['name'][box_mask]

            difficulty.append(info['difficulty'][box_mask])
            score.append(np.ones(num_boxes))
            if fake_gt_infos:
                info['gt_boxes_lidar'] = boxes3d_kitti_fakelidar_to_lidar(info['gt_boxes_lidar'])

            if info['gt_boxes_lidar'].shape[-1] == 9:
                boxes3d.append(info['gt_boxes_lidar'][box_mask][:, 0:7])
            else:
                boxes3d.append(info['gt_boxes_lidar'][box_mask])
        else:
            num_boxes = len(info['boxes_lidar'])
            difficulty.append([0] * num_boxes)
            score.append(info['score'])
            boxes3d.append(np.array(info['boxes_lidar'][:, :7]))
            box_name = info['name']
            if boxes3d[-1].shape[-1] == 9:
                boxes3d[-1] = boxes3d[-1][:, 0:7]

        obj_type += [['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist'].index(name) for i, name in enumerate(box_name)]
        frame_id.append(np.array([frame_index] * num_boxes))
        overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

    frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
    boxes3d = np.concatenate(boxes3d, axis=0)
    obj_type = np.array(obj_type).reshape(-1)
    score = np.concatenate(score).reshape(-1)
    overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
    difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

    boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

    return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

def mask_by_distance(distance_thresh, boxes_3d, *args):
    mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
    boxes_3d = boxes_3d[mask]
    ret_ans = [boxes_3d]
    for arg in args:
        ret_ans.append(arg[mask])

    return tuple(ret_ans)

def waymo_evaluation(prediction_infos, gt_infos, class_name, sep, distance_thresh=100, fake_gt_infos=True):
    print('Start the waymo evaluation...')
    assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

    pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = generate_waymo_type_results(
        prediction_infos, class_name, is_gt=False
    )
    gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = generate_waymo_type_results(
        gt_infos, class_name, is_gt=True, fake_gt_infos=fake_gt_infos
    )

    pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = mask_by_distance(
        distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
    )
    gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = mask_by_distance(
        distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty
    )
    
    print('Number: (pd, %d) vs. (gt, %d)' % (len(pd_boxes3d), len(gt_boxes3d)))
    print('Level 1: %d, Level 2: %d' % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum()))

    if pd_score.max() > 1:
        # assert pd_score.max() <= 1.0, 'Waymo evaluation only supports normalized scores'
        pd_score = 1 / (1 + np.exp(-pd_score))
        print('Warning: Waymo evaluation only supports normalized scores')

    if sep:
        pd = split([pd_frameid, pd_type], [pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz])
        gt = split([gt_frameid, gt_type], [gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty])
        
        classes = set(pd.keys()).intersection(set(gt.keys()))
        #classes = gt.keys()
        aps = {f"{i[0]}_{['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist'][i[1]]}_L2_AP":calculate_map(*pd[i],*gt[i]) for i in classes}
        
    else:
        pd = split([pd_type], [pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz])
        gt = split([gt_type], [gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty])
	
        #classes=set(pd.keys())
        classes = set(pd.keys()).intersection(set(gt.keys()))
        classes = gt.keys()
        aps = {f"{['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist'][i[0]]}_L2_AP":calculate_map(*pd[i],*gt[i]) for i in classes}
        #aps = {f"{['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist'][i[0]]}_L1_AP":calculate_map(*pd[i],*gt[(i[0],1)]) for i in classes}

    return aps

def split(criteria, lists, filter=lambda x: True):
    index = list(zip(*criteria))
    dic = {k:[] for k in index}
    data = list(zip(*lists))
    for i,v in list(zip(index, data)):
        dic[i].append(v)
    return {k:list(zip(*v)) for k,v in dic.items() if filter(k)}

@jit(nopython=True)
def calculate_iou(b1, b2):
    x1, y1, z1, dx1, dy1, dz1, _ = b1
    x2, y2, z2, dx2, dy2, dz2, _ = b2
    
    x1i = max(x1 - 0.5*dx1, x2 - 0.5*dx2)
    y1i = max(y1 - 0.5*dy1, y2 - 0.5*dy2)
    z1i = max(z1 - 0.5*dz1, z2 - 0.5*dz2)
    x2i = min(x1 + 0.5*dx1, x2 + 0.5*dx2)
    y2i = min(y1 + 0.5*dy1, y2 + 0.5*dy2)
    z2i = min(z1 + 0.5*dz1, z2 + 0.5*dz2)
    
    if (x2i <= x1i or y2i <= y1i or z2i <= z1i):
        return 0

    int_volume = (x2i - x1i) * (y2i - y1i) * (z2i - z1i)
    union_volume = dx1 * dy1 * dz1 + dx2 * dy2 * dz2 - int_volume

    return int_volume / union_volume

def calculate_per_frame_truth(pd, gt, threshold):
    if len(pd) == 0:
        return [], []
    pd, pd_score = pd
    gt = gt[0]
    pd, pd_score = zip(*sorted(zip(pd, pd_score), key=lambda x: x[1], reverse=True))
    tf = np.zeros(len(pd))
    for gti in gt:
        for i, pdi in enumerate(pd):
            if (calculate_iou(pdi, gti) > threshold):
                tf[i] = 1
                break
    return tf, pd_score

def calculate_per_class_pr(pd, gt, threshold):
    if pd == 0 or gt == 0:
        return 0
    pd_frameid, pd_boxes3d, pd_score = pd
    gt_frameid, gt_boxes3d = gt
    pd = split([pd_frameid], [pd_boxes3d, pd_score])
    gt = split([gt_frameid], [gt_boxes3d])
    frames = set(pd.keys()).union(set(gt.keys()))
    ta = []
    confidence = []
    for i in frames:
        tf, pd_score = calculate_per_frame_truth(pd.get(i, []), gt.get(i, [[]]), threshold)
        ta.extend(tf)
        confidence.extend(pd_score)
    ta, _ = zip(*sorted(zip(ta, confidence), key=lambda x: x[1], reverse=True))
    total = len(gt_boxes3d)
    if total == 0:
        return [], []
    run = 0
    p = [1]
    r = [0]
    for i, v in enumerate(ta, 1):
        run += v
        p.append(run/i)
        r.append(run/total)
    p.append(0)
    r.append(1)
    #flattening
    last = p[len(p)-1]
    for i in range(len(p))[-2::-1]:
        lr = r[i+1]
        while (lr - r[i] > 0.05 + 1e-6):
            lr -= 0.05
            p.insert(i+1, last)
            r.insert(i+1, lr)
        if p[i] > last:
            last = p[i]
        else:
            p[i] = last
    return p, r

def auc(p, r):
    #p,r = list(zip(*sorted(zip(p,r), key=lambda x: (x[1],-x[0]))))
    area = 0
    for i in range(1, len(r)):
        #area += p[i] * (r[i] - r[i-1])
        area += (p[i-1] + p[i]) / 2 * (r[i] - r[i-1])
    return area
    
def calculate_map(pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz,
                  gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty,
                  class_thresholds = [0.0,0.7,0.5,0.5,0.5]):
    pd = split([pd_type], [pd_frameid, pd_boxes3d, pd_score])
    gt = split([gt_type], [gt_frameid, gt_boxes3d])
    classes = set(pd.keys()).union(set(gt.keys()))
    return sum([auc(*calculate_per_class_pr(pd.get(i, 0), gt.get(i, 0), class_thresholds[i[0]])) for i in classes]) / len(classes)


def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--predictions', type=str, default=None, help='pickle file with predictions')
    parser.add_argument('--config', type=str, default=None, help='yaml config file for the model')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--output', type=str, default=None, help='output txt file')
    parser.add_argument('--per_frame', type=str, default="False", help='per frame analysis')
    parser.add_argument('--start', type=int, default=0, help='start frame of ground truth (0-indexed)')
    args = parser.parse_args()
    
    time0 = time.time()
    pd = pickle.load(open(args.predictions, 'rb'))
    cfg_from_yaml_file(args.config, cfg)
    cfg.TAG = Path(args.config).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.config.split('/')[1:-1])
    
    dataset = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=args.batch, dist=False, training=False)[1].dataset
    class_names = dataset.class_names
    gt = [copy.deepcopy(info['annos']) for info in dataset.infos]
    gt = gt[args.start:args.start+len(pd)]
    
    rt = waymo_evaluation(pd, gt, class_names, args.per_frame == "True", 1000, fake_gt_infos=False)
    rt = dict(sorted(rt.items()))
    open(args.output, "w").write(str(rt))
    time1 = time.time()
    print('Total profiling time', (time1 - time0))
if __name__ == '__main__':
    main()
