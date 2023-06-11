# Official Implementation: "AutoTomo: Learning-Based Traffic Estimator Incorporating Network Tomographyy"

This is the github repoistory for an accurate and low-cost traffic estimation approach implemented in pytorch.

### Paper
AutoTomo: Learning-Based Traffic Estimator Incorporating Network Tomography

## Abstract

> Estimating the Traffic Matrix (TM) is a critical yet resource-intensive process in network management. With the advent of deep learning models, we now have the potential to learn the inverse mapping from link loads to origin-destination (OD) flows more efficiently and accurately. However, a significant hurdle is that all current learning-based techniques necessitate a training dataset covering a comprehensive TM for a specific duration. This requirement is often unfeasible in practical scenarios. This paper addresses this complex learning challenge, specifically when dealing with incomplete and biased TM data. 

> Our initial approach involves parameterizing the unidentified flows, thereby transforming this problem of target-deficient learning into an empirical optimization problem that integrates tomography constraints. Following this, we introduce AutoTomo, a learning-based architecture designed to optimize both the inverse mapping and the unexplored flows during the model’s training phase. We also propose an innovative observation selection algorithm, which aids network operators in gathering the most insightful measurements with limited device resources. We evaluate AutoTomo with two public traffic datasets Abilene
and GEANT. The results reveal that AutoTomo outperforms four state-of-the-art learning-based TM estimation techniques. With complete training data, AutoTomo enhances the accuracy of the most efficient method by 15%, while it shows an improvement between 30% to 53% with incomplete training data. Furthermore, AutoTomo exhibits rapid testing speed, making it a viable tool for real-time TM estimation.

## Requirements

This project has been developed and tested in Python 3.7.13 and requires the following libraries:

- Numpy==1.21.6
- Pandas==1.3.5
- Matplotlib==3.6.0

## Framework

- Pytorch==1.12.0

## Datasets

- Abilene Dataset
- G´EANT Dataset

## Methods

- AutoTomo
- BPNN
- DBN
- VAE &nbsp;&nbsp; details in https://github.com/MikeKalnt/VAE-TME
- FGSR &nbsp; details in https://github.com/udellgroup/Codes-of-FGSR-for-effecient-low-rank-matrix-recovery

## File structure:

- data_utils.py - preprocess CSV file and custom dataset loader for experimentation
- main.py - main function with args settings
- plot_utils.py - helper functions for generating figures 
- train_utils.py - code for earlystopping, training and testing
- net_params.py - network architectures of AutoTomo, AutoTomo-os, BPNN and DBN
- Data/Abilene/ - CSV files of remote martrix and traffic matrix sampled from Abilene dataset
- Data/Geant/ - CSV files of remote martrix and traffic matrix sampled from G´EANT dataset

## Args:

{
"model": ["autotomo", "autotomo-os", "bpnn", "dbn"],
"unknown_r": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
"mode": ["all", "unobserved"],
"loss_func": ["mse", "l1norm"],
"patience": 50,
"epoch": 500,
"batch_size": 32,
"lr": 0.001,
"dataset": ["abilene", "geant"]
}

## How to use

Simple usage:
```bash
python main.py --model autotomo --unknown_r 0.1
```

Example usage:
```bash
python main.py --model autotomo --unknown_r 0.1 --dataset geant --loss_func l1norm --mode unobserved --epoch 1500
```

## Licence

Distributed under the MIT License. See [LICENCE](https://github.com/Y-debug-sys/AutoTomo/blob/main/LICENSE) for more information.
