# Agile3D: Adaptive Contention- and Content-Aware 3D Object Detection for Embedded GPUs

Authors: xxx

Efficient 3D perception is critical for autonomous systems like self-driving vehicles and drones to operate safely in dynamic environments. Accurate 3D object detection from LiDAR data faces challenges due to the irregularity and high volume of point clouds, inference latency variability from contention and content dependence, and embedded hardware constraints. Balancing accuracy and latency under dynamical conditions is crucial, yet existing frameworks like Chanakya [NeurIPS ’23], LiteReconfig [EuroSys ’22], and AdaScale [MLSys ’19] struggle with the unique demands of 3D detection. We present Agile3D, the first adaptive 3D system to integrate a cross-model Multi-branch Execution Framework (MEF) and a Contention- and Content-Aware RL-based controller (CARL). CARL dynamically selects the optimal execution branch using five novel MEF control knobs: partitioning format, spatial resolution, spatial encoding, 3D feature extractors, and detection heads. CARL employs a dual-stage optimization strategy: Supervised pretraining for robust initial learning and Direct Preference Optimization (DPO) for fine-tuning without manually tuned rewards, inspired by techniques for training large language models. Comprehensive evaluations show that Agile3D achieves state-of-the-art performance, maintaining high accuracy across varying hardware contention levels and latency budgets of 100-500 ms. On NVIDIA Orin and Xavier GPUs, it consistently leads the Pareto frontier, outperforming existing methods for robust, efficient 3D object detection.

## 1. Setup the Environment
We provide an embedded GPU and a desktop GPU for you to evaluate on.   
Please find the IP address below and the ssh-key in the Artifact Appendix.  

### 1.1. Install ZeroTier
Becaus of university IT security restrictions, we use ZeroTier to give external users access.  
Install ZeroTier on your local computer following the instructions here: https://www.zerotier.com/download/  

### 1.2. Join the Network 
Join the network we setup for Artifact Evaluation with ID: a09acf02337ca32e  
We will authorize your local device in 24 hours.
<!-- On Linux:
```
# Send us the address, e.g. '312eec1a24', on HotCRP for authorization
sudo zerotier-cli status
> 200 info 312eec1a24 1.14.2 ONLINE
```
On Windows:  
![alt text](image-3.png) -->

### 1.3. Access the embedded GPU for running experiment and the desktop GPU for evaluation
Then you can use ssh to access our embedded GPU for running experiment and desktop GPU for evaluation.
```
# ssh into the Orin embedded GPU
ssh -i mobisys2025.pem agile3d@172.30.53.226
# ssh into the desktop GPU
ssh -i mobisys2025.pem agile3d@172.30.166.233

# we already have the docker environment running on both the embedded and desktop GPUs
# On the Orin embedded GPU
docker exec -it d94aaeae45f7 /bin/bash
# On the desktop GPU
docker exec -it mobisys2025 /bin/bash
```

## 2. Run the Experiments and Evaluations
### 2.1. Activate Python Environment
If you use the board that we provide, you may use the conda environment that we provide,
```
conda activate agile3d
```
Otherwise, check our [Installation Guide](docs/INSTALL.md).


## 3. Run the Experiments and Evaluations
### 3.1. Experiment (E1)
[Key accuracy and latency performance of AGILE3D ] [40 human-minutes + 4 compute-hours]: we will run AGILE3D on the NVIDIA Jetson Orin board and examine the key accuracy and latency performance of it. Expected accuracy and latency on Orin are [71.72%, 374 ms], [70.98%, 430 ms], [70.03%, 450 ms], and [68.72%, 470 ms] under four different contention levels. Run the following commands on Orin and GPU server:
```
# This part not necessary as we already put a copy of results on the server
# Ssh into Orin
$ ssh -i mobisys2025.pem agile3d@172.30.53.226
# Get into the docker env
$ docker exec -it d94aaeae45f7 /bin/bash
# On Orin 
$ conda activate agile3d
(agile3d) $ cd /home/data/agile3d
(agile3d) $ bash experiment_1.sh
# Sync the results to the GPU server
(agile3d) $ bash results_sync.sh


# Then log into the desktop GPU for evaluation. (Recommend start from here)
# Ssh into the desktop GPU
$ ssh -i mobisys2025.pem agile3d@172.30.166.233
# Get into the docker env
$ docker exec -it mobisys2025 /bin/bash
# On GPU Server
# The waymo evaluation probably takes a long time. We provide a copy of results on the server.
# You can comment the waymo evaluation in the bash script and only print the results.
$ conda activate agile3d
(agile3d) $ cd /home/data/agile3d
(agile3d) $ bash eval_experiment_1.sh
```
The evaluation script will print the latency and accuracy results.

### 3.2. Experiment (E2)
[The low switching overhead of AGILE3D ] [20 human-minutes + 6 compute-hours]: we will run all branches in AGILE3D on Orin, switching from one branch to another, and examine the switching overhead (latency increase in the first frame after swicth). The expected mean switching overhead is under 2 ms.
On Orin, run the following command:
```
# Ssh into Orin
$ ssh -i mobisys2025.pem agile3d@172.30.53.226
# Get into the docker env
$ docker exec -it d94aaeae45f7 /bin/bash
# On Orin
# The switching overhead will be printed out during the experiment
$ conda activate agile3d
(agile3d) $ cd /home/data/agile3d
(agile3d) $ bash experiment_2.sh
```
The evaluation script will print the switching overhead results.


## 3.3. Experiment (E3)
[The accuracy and latency improvement of AGILE3D over variants of static SOTA models] [60 human-minutes + 10 compute-hours]: we will run AGILE3D on Orin and examine the latency performance of it. Expected mean latency of AGILE3D is from 100 to 350 ms. Those of PV-RCNN, DSVT-Voxel, and DSVTPillar are 850 ms, 460 ms, and 350 ms. So AGILE3D achieves both faster speed and higher accuracy than static SOTA baselines. Run the following commands on Orin and GPU server:
```
# Ssh into Orin
$ ssh -i mobisys2025.pem agile3d@172.30.53.226
# Get into the docker env
$ docker exec -it d94aaeae45f7 /bin/bash
# On Orin
# This part is not necessary as we already put a copy of results on the server
$ conda activate agile3d
(agile3d) $ cd /home/data/agile3d
(agile3d) $ bash experiment_3.sh
# Sync the results to the GPU server
(agile3d) $ bash results_sync.sh

# Then log into the desktop GPU for evaluation. (Recommend start from here)
# Ssh into the desktop GPU
$ ssh -i mobisys2025.pem agile3d@172.30.166.233
# Get into the docker env
$ docker exec -it mobisys2025 /bin/bash
# On GPU Server
$ . activate_agile3d_env.sh
(agile3d) $ cd /home/data/agile3d
(agile3d) $ bash eval_experiment_3.sh
```
The evaluation script will print the switching overhead results.

## License

`Agile3D` is released under the [CC BY-NC-ND 4.0 license](LICENSE).

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

