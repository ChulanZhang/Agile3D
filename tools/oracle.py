import re
import os
import pickle
import numpy as np
# Oracle controller
branches = ['centerpoint_dyn_pillar024_4x',
            'centerpoint_dyn_pillar028_4x',
            'centerpoint_dyn_pillar032_4x',
            'centerpoint_dyn_pillar036_4x',
            'centerpoint_dyn_pillar040_4x',
            'centerpoint_dyn_pillar044_4x',
            'centerpoint_dyn_pillar048_4x',
            'centerpoint_dyn_pillar052_4x',
            'centerpoint_without_resnet_dyn_voxel100',
            'centerpoint_without_resnet_dyn_voxel150',
            'centerpoint_without_resnet_dyn_voxel200',
            'centerpoint_without_resnet_dyn_voxel250',
            'centerpoint_without_resnet_dyn_voxel300',
            'centerpoint_without_resnet_dyn_voxel350',
            'centerpoint_without_resnet_dyn_voxel400',
            'centerpoint_without_resnet_dyn_voxel450',
            'dsvt_sampled_pillar020',
            'dsvt_sampled_pillar030',
            'dsvt_sampled_pillar040',
            'dsvt_sampled_pillar050',
            'dsvt_sampled_pillar060',
            'dsvt_sampled_pillar070',
            'dsvt_sampled_pillar080',
            'dsvt_sampled_pillar090',
            'dsvt_sampled_pillar100',
            'dsvt_sampled_pillar110',
            'dsvt_sampled_pillar120',
            'dsvt_sampled_pillar130',
            'dsvt_sampled_voxel020',
            'dsvt_sampled_voxel030',
            'dsvt_sampled_voxel040',
            'dsvt_sampled_voxel050',
            'dsvt_sampled_voxel060',
            'dsvt_sampled_voxel070',
            'dsvt_sampled_voxel080',
            'dsvt_sampled_voxel090',
            'dsvt_sampled_voxel100',
            'dsvt_sampled_voxel110',
            'dsvt_sampled_voxel120',
            'dsvt_sampled_voxel130']

data_root = '../output/waymo_new_profiling'

# latency profiles
print('Loading branch latency for test samples...')
lat_dir = os.path.join(data_root, 'lat/test')
branch_lats = []
for b in branches:
    lat_path = os.path.join(lat_dir, b + '_lat.pkl')
    lat = pickle.load(open(lat_path, 'rb'))
    branch_lats.append(np.array(lat))
branch_lats = np.stack(branch_lats, axis=-1)    # (#samples, #branches)

# accuracy profiles in waymo format
print('Loading branch accuracy for test samples...')
branch_accs = np.load('../output/waymo_new_profiling/per_frame_l2_acc_test.npy', allow_pickle=True)

# load detection profiles
print('Loading branch detection results for test samples...')
det_dir = os.path.join(data_root, 'det/test')
branch_profiles = []
for b in branches:
    det_path = os.path.join(det_dir, b + '_det.pkl')
    det = pickle.load(open(det_path, 'rb'))
    branch_profiles.append(det)
branch_profiles = np.stack(branch_profiles, axis=-1)    # (#samples, #branches)

# latency budgets
board = 'orin'
slo_list = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
# slo_list = (1000, 2000)
schd_overhead = 0

# SWITCH BETWEEN LATENCY PREDICTORS AND THRESHOLDS
print('Running virtual branch scheduling...')
det_results, lat_results = dict(), dict()
for slo in slo_list:
    det_results[slo] = []
    lat_results[slo] = []

# k is the number of frames
for k in range(len(branch_lats)):
    if (k + 1) % 500 == 0:
        print(f'{(k + 1):04d} done!')

    # for each frame, load the lat and acc of 20 branches
    lat, acc = branch_lats[k], branch_accs[k]

    for slo in slo_list:
        # Filter valid branches according to the latency slo
        valid_branches = np.nonzero(lat + schd_overhead <= slo)[0]
        # If there is no valid branch, choose the fastest branch
        if len(valid_branches) == 0:
            valid_branches = [lat.argmin()]
        # Select the branch come with the highest acc
        # If multiple branches have the same highest acc, choose the fastest branch
        sorted_indices = np.lexsort((lat[valid_branches], -acc[valid_branches]))
        b = valid_branches[sorted_indices[0]]
        lat_results[slo].append(lat[b])

        # load detection profile
        det = branch_profiles[k][b]
        det_results[slo].append(det)

print('Saving detection results...')
out_dir = '../output/oracle/oracle'
os.makedirs(out_dir, exist_ok=True)
for slo in slo_list:
    out_path = os.path.join(out_dir, f'{board}_slo{slo:d}_oracle.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(det_results[slo], f)

    lat = np.array(lat_results[slo]).mean()
    print(f"Latency [SLO: {slo:d}]: {lat:.2f}s")