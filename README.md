# Agile3D: Adaptive Contention- and Content-Aware 3D Object Detection for Embedded GPUs

Authors: xxx

Efficient 3D perception is critical for autonomous systems like self-driving vehicles and drones to operate safely in dynamic environments. Accurate 3D object detection from LiDAR data faces challenges due to the irregularity and high volume of point clouds, inference latency variability from contention and content dependence, and embedded hardware constraints. Balancing accuracy and latency under dynamical conditions is crucial, yet existing frameworks like Chanakya [NeurIPS ’24], LiteReconfig [EuroSys ’22], and AdaScale [MLSys ’19] struggle with the unique demands of 3D detection. We present Agile3D, the first adaptive 3D system to integrate a cross-model Multi-branch Execution Framework (MEF) and a Contention- and Content-Aware RL-based controller (CARL). CARL dynamically selects the optimal execution branch using five novel MEF control knobs: partitioning format, spatial resolution, spatial encoding, 3D feature extractors, and detection heads. CARL employs a dual-stage optimization strategy: Supervised pretraining for robust initial learning and Direct Preference Optimization (DPO) for fine-tuning without manually tuned rewards, inspired by techniques for training large language models. Comprehensive evaluations show that Agile3D achieves state-of-the-art performance, maintaining high accuracy across varying hardware contention levels and latency budgets of 100-500 ms. On NVIDIA Orin and Xavier GPUs, it consistently leads the Pareto frontier, outperforming existing methods for robust, efficient 3D object detection.

## Setup the Evaluation
An NVIDIA Jetson Orin board. We provide a board for you to evaluate on. Please find the ip address and key in the Artifact Apeendix.
```
# ssh into the Orin board
ssh -i mobisys2025.pem agile3d@$ORIN_IP
# list the tmux sessions
tmux ls 
# we already have the docker env running on the board
tmux a -t mobisys2025

# ssh into the GPU server
ssh -i mobisys2025.pem agile3d@$SERVER_IP
# list the tmux sessions
tmux ls 
tmux a -t mobisys2025
```

## Installation
If you use the board that we provide, you may use the conda environment that we provide,
```
conda activate agile3d
```
Otherwise, check our [Installation Guide](docs/INSTALL.md).

## Code, Models, and Datasets
If you use the board that we provide, the code, models, and datasets are already placed in the following file tree,
```
/home/agile3d/agile3d #source code directory
/home/agile3d/agile3d/checkpoints  # all the trained models
/home/agile3d/agile3d/data/waymo  # the path to Waymo dataset
```

# Experiments
## Experiment (E1)
[Key accuracy and latency performance of \name] [40 human-minutes + 4 compute-hours]: we will run \name on the NVIDIA Jetson Orin board and examine the key accuracy and latency performance of it. Expected accuracy and latency on Orin are [71.72\%, 374 ms], [70.98\%, 430 ms], [70.03\%, 450 ms], and [68.72\%, 470 ms] under four different contention levels. 

Run the following commands on Orin and GPU server,
```
# On Orin
$ conda activate agile3d
(agile3d) $ cd ~/agile3d
(agile3d) $ bash experiment_1.sh
# Sync the results to the GPU server
(agile3d) $ bash results_sync.sh

# On GPU Server
$ . activate_agile3d_env.sh
(agile3d) $ cd ~/agile3d
(agile3d) $ bash eval_experiment_1.sh
```
The results will be written to ``agile3d/output/e1_results``. We have saved a copy of these files in ``agile3d/output/e1_results_saved``, and use ``bash eval_experiment_1.sh`` to compute the accuracy and latency from these results files. One may replace the filenames by those in the online execution.

## Experiment (E2)
[The low switching overhead of \name] [20 human-minutes + 6 compute-hours]: we will run all branches in \name on Orin, switching from one branch to another, and examine the switching overhead (latency increase in the first frame after swicth). The expected mean switching overhead is under 2 ms.

On Orin, run the following commands,
```
# On Orin
$ conda activate agile3d
(agile3d) $ cd ~/agile3d
(agile3d) $ bash experiment_2.sh
```
The results will be written to ``agile3d/output/e2_results``.

## Experiment (E3)
[The accuracy and latency improvement of \name over variants of static SOTA models] [60 human-minutes + 10 compute-hours]: we will run \name on Orin and examine the latency performance of it. Expected mean latency of \name is from 50 to 350 ms. Those of PV-RCNN, DSVT-Voxel, and DSVT-Pillar are 850 ms, 460 ms, and 350 ms. So \name achieves both faster speed and higher accuracy than static SOTA baselines. 

Run the following commands on Orin and GPU server,
```
# On Orin
$ conda activate agile3d
(agile3d) $ cd ~/agile3d
(agile3d) $ bash experiment_3.sh
# Sync the results to the GPU server
(agile3d) $ bash results_sync.sh

# On GPU Server
$ . activate_agile3d_env.sh
(agile3d) $ cd ~/agile3d
(agile3d) $ bash eval_experiment_3.sh
```
The results will be written to \url{agile3d/output/e3_results}.


## License

`Agile3d` is released under the [CC BY-NC-ND 4.0 license](LICENSE).

## Acknowledgement
xxx


## Citation 
If you find this project useful in your research, please consider cite:

```
@misc{agil3d,
    title={Agile3D: Adaptive Contention- and Content-Aware 3D Object Detection for Embedded GPUs},
    author={xxx},
    howpublished = {xxx},
    year={2025}
}
```

