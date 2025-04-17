import numpy as np
import pickle
import ast

branches = ['centerpoint_without_resnet_dyn_voxel100', 'dsvt_sampled_pillar066', 'dsvt_sampled_pillar048', 
            'dsvt_sampled_voxel058', 'dsvt_sampled_voxel048', 'dsvt_sampled_voxel040', 'dsvt_sampled_voxel038']

def load_file_to_dict(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        key, value = content.split(" : ", 1)
        data_dict = ast.literal_eval(value)  # Convert string to dictionary
        return {key.strip(): data_dict}

for branch in branches:
    # Load latency results
    lat_path = f'/home/data/profiling_results/lat/test/{branch}_lat.pkl'
    with open(lat_path, 'rb') as f:
        lat_results = pickle.load(f)
    
    # Load accuracy results
    acc_path = f'/home/data/agile3d/output/exp3/{branch}.txt'
    data_dict = load_file_to_dict(acc_path)
    
    # Calculate AP metrics
    ap_1 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'][0]) + 
            float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'][0]) + 
            float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'][0])) / 3
    
    ap_2 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'][0]) + 
            float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'][0]) + 
            float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'][0])) / 3
    
    # Print results
    print(f"{branch}\t{lat_results:.2f}\t\t{ap_1:.2f}\t{ap_2:.2f}")