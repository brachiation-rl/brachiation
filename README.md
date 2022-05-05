# Learning to Brachiate via Simplified Model Imitation

![Python](https://img.shields.io/badge/Python->=3.8-Blue?logo=python)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.9.0-Red?logo=pytorch)

This repo is the codebase for the SIGGRAPH 2022 conference paper with the title above. 
Please find the paper and demo at our project website https://brachiation-rl.github.io/brachiation/.

<img src="docs/static/assets/teaser.gif">

## Prerequisites

* Linux or macOS
* Python 3.8 or newer

### Install Requirements

Download and install custom PyBullet build
```bash
git clone git@github.com:belinghy/bullet3.git
pip install ./bullet3
```

Install required packages
```bash
git@github.com:brachiation-rl/brachiation.git
cd brachiation
pip install -r requirements.txt
```

## Quick Start

Test installation is complete by running passive forward simulation
```bash
python run_full.py --mode test
```

This repo contains pretrained models and examples of generated trajectories.
Run and visualize the pretrained full model controller
```bash
python run_full.py --mode play --net data/best_full.pt
```

Run and visualize simplified model controller
```bash
python run_simplified.py --mode play --net data/best_simple.pt
```

Run and visualize full model planning.
This mode is slow on when simulating on CPU; if needed, reduce the number of parallel simulations on [L61](https://github.com/brachiation-rl/brachiation/blob/b080341bb2c7c1f0fe603c9819db6aa20fac59f6/run_full.py#L461) in run_full.py.
```bash
python run_full.py --mode plan --net data/best_full.pt
```

## Training

Training simplified and full model from scratch
```bash
# Step 1: Train simplified and dump trajectories in data/trajectories/
python run_simplified.py --mode train
python run_simplified.py --mode dump --net <saved-model-file>

# Step 2: Train full model (uses previously saved trajectories)
python run_full.py --mode train
```

## Citation

If you use this code for your research, please cite our paper.
The BibTeX entry will be posted after paper becomes available on the ACM website.