# RL scheduler for 3d point cloud detection

This codebase holds different RL methods to train a RL scheduler for the 3D point cloud system

## training with contention
### supervised training
```
CUDA_VISIBLE_DEVICES=0 python train_supervised.py --name waymo_supervised_contention_slo500_accumulate16_lr1e3_range_5_10 \
--opt config/st/waymo_supervised_contention_slo500_accumulate16_lr1e3_range_5_10.yaml
```

### dpo training
```
python train_dpo.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10 \
--opt config/dpo/waymo_dpo_contention_slo500_accumulate16_range_5_10.yaml \
--ref-model-path ./experiments/waymo_supervised_contention_slo500_accumulate16_lr1e3_range_5_10
```
### The evaluation

```
cd /depot/schaterj/data/3d/work_dir/zhuoming_temp/3d_scheduler
. activate_env_rl.sh

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10 --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl00.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10  \
--interval-schedule --notfilter --contention-level 0.0 --lantency-file-path /depot/schaterj/data/3d/waymo_results/waymo_new_profiling/latency_test_55b_c00.npy

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10 --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl20.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10  \
--interval-schedule --notfilter --contention-level 0.2 --lantency-file-path /depot/schaterj/data/3d/waymo_results/waymo_new_profiling/latency_test_55b_c20.npy

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10 --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl50.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10  \
--interval-schedule --notfilter --contention-level 0.5 --lantency-file-path /depot/schaterj/data/3d/waymo_results/waymo_new_profiling/latency_test_55b_c50.npy


python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10 --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl90.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10  \
--interval-schedule --notfilter --contention-level 0.9 --lantency-file-path /depot/schaterj/data/3d/waymo_results/waymo_new_profiling/latency_test_55b_c90.npy



cd ~
. activate_env_tf_cpu.sh
cd /depot/schaterj/data/3d/work_dir/adaptive-3d-openpcdet-baseline/tools

python eval.py --predictions /depot/schaterj/data/3d/work_dir/zhuoming_temp/3d_scheduler/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10/e9_cl00.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl00.txt --start 0

python eval.py --predictions /depot/schaterj/data/3d/work_dir/zhuoming_temp/3d_scheduler/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10/e9_cl20.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl20.txt --start 0

python eval.py --predictions /depot/schaterj/data/3d/work_dir/zhuoming_temp/3d_scheduler/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10/e9_cl50.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl50.txt --start 0

python eval.py --predictions /depot/schaterj/data/3d/work_dir/zhuoming_temp/3d_scheduler/experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10/e9_cl90.pkl \
--config cfgs/waymo_models/centerpoint_dyn_pillar060_4x.yaml --output waymo_dpo_contention_slo500_accumulate16_range_5_10_e9_cl90.txt --start 0
```