cd tools

python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl00.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/carl/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl00.txt --start 0

python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl20.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/carl/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl20.txt --start 0

python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl50.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/carl/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl50.txt --start 0

python eval.py --predictions /home/data/agile3d/carl/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare/e9_cl90.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output /home/data/agile3d/carl/waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl90.txt --start 0


# Calculate the final metric
cd /home/data/agile3d/carl
python tools/map_res_extract_v2.py --path waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl00.txt

python tools/map_res_extract_v2.py --path waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl20.txt

python tools/map_res_extract_v2.py --path waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl50.txt

python tools/map_res_extract_v2.py --path waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl90.txt
