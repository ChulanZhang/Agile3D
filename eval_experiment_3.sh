# Not necessary to run this script, the results are saved in /home/data/agile3d/output/exp3/
cd /home/data/agile3d/tools
python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/centerpoint_without_resnet_dyn_voxel100_det.pkl \
--config cfgs/waymo_models/centerpoint_without_resnet_dyn_voxel100.yaml --output /home/data/agile3d/output/exp3/centerpoint_without_resnet_dyn_voxel100.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_pillar066_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_pillar066.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_pillar066.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_pillar048_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_pillar048.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_pillar048.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_voxel058_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_voxel058.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_voxel058.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_voxel048_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_voxel048.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_voxel048.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_voxel040_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_voxel040.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_voxel040.txt --start 0

python eval.py --predictions /home/data/agile3d/output/profiling_results/det/test/dsvt_sampled_voxel038_det.pkl \
--config cfgs/waymo_models/dsvt_sampled_voxel038.yaml --output /home/data/agile3d/output/exp3/dsvt_sampled_voxel038.txt --start 0

# All the results are saved in /home/data/agile3d/output/exp3/
# You can comment out the above commands and run the following command to see the final metric
# Print results
cd /home/data/agile3d/tools
python exp3_print_results.py

