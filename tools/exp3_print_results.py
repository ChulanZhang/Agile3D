import numpy as np
import pickle
import ast

# Define branches for experiment 3
pareto_branches = [
    'centerpoint_without_resnet_dyn_voxel100', 
    'dsvt_sampled_pillar066', 
    'dsvt_sampled_pillar048', 
    'dsvt_sampled_voxel058', 
    'dsvt_sampled_voxel048', 
    'dsvt_sampled_voxel040', 
    'dsvt_sampled_voxel038'
]

baselines = [
    'centerpoint_pillar_1x',
    'centerpoint_without_resnet',
    'second',
    'PartA2',
    'pointpillar_1x',
    'pv_rcnn',
    'dsvt_sampled_voxel032',
    'dsvt_sampled_pillar032',
]

# Define shorter display names for branches
branch_display_names = {
    'centerpoint_without_resnet_dyn_voxel100': 'Agile3D-CP-Voxel100',
    'dsvt_sampled_pillar066': 'Agile3D-DSVT-Pillar066',
    'dsvt_sampled_pillar048': 'Agile3D-DSVT-Pillar048',
    'dsvt_sampled_voxel058': 'Agile3D-DSVT-Voxel058',
    'dsvt_sampled_voxel048': 'Agile3D-DSVT-Voxel048',
    'dsvt_sampled_voxel040': 'Agile3D-DSVT-Voxel040',
    'dsvt_sampled_voxel038': 'Agile3D-DSVT-Voxel038',
    'dsvt_sampled_voxel032': 'DSVT-Voxel',
    'dsvt_sampled_pillar032': 'DSVT-Pillar',
    'centerpoint_pillar_1x': 'CenterPoint-Pillar',
    'centerpoint_without_resnet': 'CenterPoint-Voxel',
    'second': 'SECOND', 
    'PartA2': 'PartA2',
    'pointpillar_1x': 'PointPillar',
    'pv_rcnn': 'PV-RCNN'
}

def load_latency_data(file_path):
    """Load latency data from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            lat_results = pickle.load(f)
            return np.mean(lat_results)
    except Exception as e:
        print(f"Error loading latency data from {file_path}: {e}")
        return float('nan')

def load_accuracy_file(file_path):
    """Load data from accuracy file"""
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            parts = content.split(" : ", 1)
            if len(parts) != 2:
                print(f"Incorrect file format: {file_path}")
                return None
            key, value = parts
            try:
                data_dict = ast.literal_eval(value)  # Convert string to dictionary
                return {key.strip(): data_dict}
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing value: {e}")
                return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def calculate_ap_metrics(data_dict):
    """Calculate AP metrics"""
    if not data_dict or '-1' not in data_dict:
        return None, None
    
    try:
        # Calculate LEVEL 1 AP (average of vehicle, pedestrian, and cyclist APs)
        ap_1 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'][0]) + 
                float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'][0]) + 
                float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'][0])) / 3
        
        # Calculate LEVEL 2 AP
        ap_2 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'][0]) + 
                float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'][0]) + 
                float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'][0])) / 3
        
        return ap_1, ap_2
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error calculating AP metrics: {e}")
        return None, None

def main():
    # Print header
    print("\nExperiment 3 Results Summary:")
    print("=" * 80)
    print("{:<20} {:<15} {:<15} {:<15}".format("Model", "Latency (ms)", "L1 mAP", "L2 mAP"))
    print("-" * 80)
    
    for branch in pareto_branches + baselines:
        # Load latency data
        lat_path = f'/home/data/profiling_results/lat/test/{branch}_lat.pkl'
        latency_val = load_latency_data(lat_path)
        
        # Load accuracy data
        acc_path = f'/home/data/agile3d/output/exp3/{branch}.txt'
        data_dict = load_accuracy_file(acc_path)
        
        if data_dict:
            # Calculate AP metrics
            ap_1, ap_2 = calculate_ap_metrics(data_dict)
        else:
            ap_1, ap_2 = float('nan'), float('nan')
        
        # Print results with branch display name
        display_name = branch_display_names.get(branch, branch)
        print("{:<20} {:<15.2f} {:<15.4f} {:<15.4f}".format(
            display_name, 
            latency_val, 
            ap_1 if ap_1 is not None else float('nan'), 
            ap_2 if ap_2 is not None else float('nan')
        ))
    
    print("=" * 80)

if __name__ == "__main__":
    main()