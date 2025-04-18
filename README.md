# Agile3D: Adaptive Contention- and Content-Aware 3D Object Detection for Embedded GPUs

Authors: xxx

Efficient 3D perception is critical for autonomous systems like self-driving vehicles and drones to operate safely in dynamic environments. Accurate 3D object detection from LiDAR data faces challenges due to the irregularity and high volume of point clouds, inference latency variability from contention and content dependence, and embedded hardware constraints. Balancing accuracy and latency under dynamical conditions is crucial, yet existing frameworks like Chanakya [NeurIPS ’23], LiteReconfig [EuroSys ’22], and AdaScale [MLSys ’19] struggle with the unique demands of 3D detection. We present Agile3D, the first adaptive 3D system to integrate a cross-model Multi-branch Execution Framework (MEF) and a Contention- and Content-Aware RL-based controller (CARL). CARL dynamically selects the optimal execution branch using five novel MEF control knobs: partitioning format, spatial resolution, spatial encoding, 3D feature extractors, and detection heads. CARL employs a dual-stage optimization strategy: Supervised pretraining for robust initial learning and Direct Preference Optimization (DPO) for fine-tuning without manually tuned rewards, inspired by techniques for training large language models. Comprehensive evaluations show that Agile3D achieves state-of-the-art performance, maintaining high accuracy across varying hardware contention levels and latency budgets of 100-500 ms. On NVIDIA Orin and Xavier GPUs, it consistently leads the Pareto frontier, outperforming existing methods for robust, efficient 3D object detection.



## Getting Started

We provide both an embedded GPU and a desktop GPU for evaluation. Access details are below.

### Accessing the Evaluation Environment

#### 1. Install ZeroTier

Due to university IT security restrictions, we use ZeroTier to provide external access.
Install ZeroTier on your local computer: [https://www.zerotier.com/download/](https://www.zerotier.com/download/)

#### 2. Join the Network

Join our ZeroTier network with ID: `a09acf02337ca32e`  
We will authorize your device within 24 hours.

#### 3. Access the GPUs

```bash
# SSH into the NVIDIA Jetson Orin (embedded GPU)
ssh -i mobisys2025.pem agile3d@172.30.53.226

# SSH into the desktop GPU for evaluation
ssh -i mobisys2025.pem agile3d@172.30.166.233

# Access the Docker environment (already running on both GPUs)
docker exec -it mobisys2025 /bin/bash
```

### Setting Up the Environment

If you use our provided systems, you can use the pre-configured conda environment:
```bash
conda activate agile3d
```

For installation on your own systems, please refer to our installation guides:
- [Server Installation Guide](INSTALL_Server.md)
- [Orin Installation Guide](INSTALL_Orin.md)

## Experiments

### Experiment 1: Performance Under Various Contention Levels
**[40 human-minutes + 4 compute-hours]**

This experiment evaluates Agile3D's accuracy and latency performance on the NVIDIA Jetson Orin under different contention levels.

Expected performance:
- Contention level 1: 71.72% accuracy, 362 ms latency
- Contention level 2: 70.98% accuracy, 415 ms latency
- Contention level 3: 70.03% accuracy, 468 ms latency
- Contention level 4: 68.72% accuracy, 476 ms latency

#### Running the experiment:

```bash
# We've already placed a copy of results on the server for your convenience
# If you want to run the experiment yourself on the GPU server:

# SSH into the GPU server and activate environment
ssh -i mobisys2025.pem agile3d@172.30.166.233
docker exec -it mobisys2025 /bin/bash
conda activate agile3d
cd /home/data/agile3d

# Run experiment_1
bash experiment_1.sh

# Evaluate experiment_1 results
bash eval_experiment_1.sh

# For fast evaluation (recommended starting point)
bash eval_experiment_1_short.sh
```

### Experiment 2: Switching Overhead
**[20 human-minutes + 6 compute-hours]**

This experiment measures the switching overhead when transitioning between different Agile3D branches.

Expected results: Mean switching overhead < 2 ms

#### Running the experiment:

```bash
# SSH into Orin and activate environment
ssh -i mobisys2025.pem agile3d@172.30.53.226
docker exec -it mobisys2025 /bin/bash
conda activate agile3d
cd /home/data/agile3d

# Run experiment_2
bash experiment_2.sh
```

### Experiment 3: Comparison with Static SOTA Models
**[60 human-minutes + 10 compute-hours]**

This experiment compares Agile3D against static state-of-the-art models in terms of latency and accuracy.

Expected latency:
- Agile3D: 100-350 ms
- PV-RCNN: 850 ms
- DSVT-Voxel: 460 ms
- DSVT-Pillar: 350 ms

#### Running the experiment:

```bash
# We've already placed a copy of results on the server for your convenience
# If you want to run the experiment yourself on Orin:

# SSH into Orin and activate environment
ssh -i mobisys2025.pem agile3d@172.30.53.226
docker exec -it mobisys2025 /bin/bash
conda activate agile3d
cd /home/data/agile3d

# Run experiment_3
bash experiment_3.sh

# For evaluation:
# SSH into GPU server and activate environment
ssh -i mobisys2025.pem agile3d@172.30.166.233
docker exec -it mobisys2025 /bin/bash
conda activate agile3d
cd /home/data/agile3d

# Evaluate experiment_3 results
bash eval_experiment_3.sh

# For fast evaluation (recommended starting point)
bash eval_experiment_3_short.sh
```

## License

Agile3D is released under the [CC BY-NC-ND 4.0 license](LICENSE).

## Acknowledgements

[Acknowledgements placeholder]

## Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{agil3d,
    title={Agile3D: Adaptive Contention- and Content-Aware 3D Object Detection for Embedded GPUs},
    author={[Author names]},
    howpublished = {[Publication venue]},
    year={2025}
}
```

