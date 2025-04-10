#! /bin/bash
# tensorflow warning
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore::FutureWarning"

cd carl

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10_spare --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl00.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare  \
--interval-schedule --notfilter --contention-level 0.0 --lantency-file-path /home/data/profiling_results/lat/latency_test_55b_c00.npy

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10_spare --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl20.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare  \
--interval-schedule --notfilter --contention-level 0.2 --lantency-file-path /home/data/profiling_results/lat/latency_test_55b_c20.npy

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10_spare --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl50.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare  \
--interval-schedule --notfilter --contention-level 0.5 --lantency-file-path /home/data/profiling_results/lat/latency_test_55b_c50.npy

python eval.py --name waymo_dpo_contention_slo500_accumulate16_range_5_10_spare --ckpt 9 \
--lantfilter --dumppickle --dumpname e9_cl90.pkl --filterthresh 500 --saveroot ./experiments/waymo_dpo_contention_slo500_accumulate16_range_5_10_spare  \
--interval-schedule --notfilter --contention-level 0.9 --lantency-file-path /home/data/profiling_results/lat/latency_test_55b_c90.npy
