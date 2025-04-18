# Not necessary to run this script, the results are saved in /home/data/agile3d/output/exp1/
# cd /home/data/agile3d/tools
# python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl00.pkl \
# --config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/output/exp1/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl00.txt --start 0

# python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl20.pkl \
# --config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/output/exp1/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl20.txt --start 0

# python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl50.pkl \
# --config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/output/exp1/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl50.txt --start 0

# python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl90.pkl \
# --config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/output/exp1/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl90.txt --start 0

# All the results are saved in /home/data/agile3d/output/exp1/
# You can comment out the above commands and run the following command to see the final metric
# Print results
cd /home/data/agile3d/tools
python exp1_print_results.py
