import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="training options")
args = parser.parse_args()

def load_file_to_dict(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        key, value = content.split(" : ", 1)
        data_dict = ast.literal_eval(value)  # Convert string to dictionary
        return {key.strip(): data_dict}

data_dict = load_file_to_dict(args.path)

ap_1 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'][0]) + float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'][0]) + float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'][0])) / 3
ap_2 = (float(data_dict['-1']['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'][0]) + float(data_dict['-1']['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'][0]) + float(data_dict['-1']['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'][0])) / 3

print('ap_1:', ap_1, 'ap_2:', ap_2)