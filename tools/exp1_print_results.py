import re
import ast
import numpy as np

# Define contention levels
branches = ['cl00', 'cl20', 'cl50', 'cl90']
# Define descriptive names for contention levels
contention_labels = {
    'cl00': 'No contention',
    'cl20': 'Light',
    'cl50': 'Moderate',
    'cl90': 'Intense'
}

def extract_latency_from_log(log_file):
    """Extract latency data from log file"""
    latencies = []
    with open(log_file, 'r') as f:
        content = f.read()
        # Use regex to find all latency values
        latency_matches = re.findall(r'final average latency: tensor\(([0-9.]+)\)', content)
        for match in latency_matches:
            latencies.append(float(match))
    return latencies

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
    # Latency log file path
    latency_log = "/home/data/agile3d/output/exp1/carl_latency_results.log"
    
    # Extract latency data
    latencies = extract_latency_from_log(latency_log)
    
    # Print header
    print("\nResults Summary:")
    print("=" * 70)
    print("{:<15} {:<15} {:<15} {:<15}".format("Contention", "Latency (ms)", "L1 AP", "L2 AP"))
    print("-" * 70)
    
    # Process each contention level
    l1_aps = []
    l2_aps = []
    
    for i, branch in enumerate(branches):
        # Accuracy file path
        acc_path = f'/home/data/agile3d/output/exp1/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_{branch}.txt'
        
        # Get latency value
        latency_val = latencies[i] if i < len(latencies) else float('nan')
        
        # Load accuracy data
        data_dict = load_accuracy_file(acc_path)
        
        if data_dict:
            # Calculate AP metrics
            ap_1, ap_2 = calculate_ap_metrics(data_dict)
            if ap_1 is not None:
                l1_aps.append(ap_1)
                l2_aps.append(ap_2)
        else:
            ap_1, ap_2 = float('nan'), float('nan')
        
        # Print results with descriptive contention level
        print("{:<15} {:<15.2f} {:<15.4f} {:<15.4f}".format(
            contention_labels[branch], 
            latency_val, 
            ap_1, 
            ap_2
        ))
    
    print("=" * 70)

if __name__ == "__main__":
    main()