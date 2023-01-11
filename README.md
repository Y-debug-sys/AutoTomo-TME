# Official Implementation: "AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography"

This is the github repoistory for an accurate and low-cost traffic estimation approach implemented in pytorch.

### Paper
AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography

## Abstract

> Traffic matrix (TM) estimation is an essential but high-cost task for network management. The state-of-the-art methods use emerging deep learning networks to learn the inverse mapping from low-cost link loads to origin-destination (OD) flows. However, all existing learning-based methods require training data covering historical instances of all OD pairs in TM to perform well. More often than not, this requirement cannot be satisfied in real-world scenarios. Thus, we must tackle the challenging learning problem with incomplete, biased data. We first parameterize the unknown flows and transform the target-deficient learning problem into an empirical optimization problem incorporating tomography constraints. Then we design a learning-based architecture named AutoTomo to optimize both inverse mapping and unknown flows through model training. Finally, we evaluate our method with two public traffic datasets Abilene and GEANT. Compared with four state-of-the-art learning-based TM estimation methods, AutoTomo improves the accuracy of the best-performed method by 15% with complete training data and 30% ∼ 53% with incomplete training data. Moreover, AutoTomo has fast testing time, enabling its practical use for real-time TM estimation.

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
- MNETME
- DBN
- VAE<br/> &nbsp;&nbsp; details in https://github.com/MikeKalnt/VAE-TME
- FGSR<br/> &nbsp;&nbsp; details in https://github.com/udellgroup/Codes-of-FGSR-for-effecient-low-rank-matrix-recovery

## File structure:

- data_process.py - preprocess CSV file and custom dataset loader for experimentation
- main.py - main function with args settings
- plot.py - helper functions for generating figures 
- utils.py - code for earlystopping, training and testing
- net_params.py - network architectures of AutoTomo, MNETME and DBN
- AbileneData/ - CSV files of remote martrix and traffic matrix sampled from Abilene dataset
- GeantData/ - CSV files of remote martrix and traffic matrix sampled from G´EANT dataset

## Example args:

{
"model": ["autotomo", "mnetme", "dbn"],
"unknown_r": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
"mode": ["all", "unobserved"],
"loss_func": ["mse", "l1norm"],
"patience": 50,
"epoch": 1000,
"batch_size": 32,
"lr": 0.01 if args.model=="mnetme" else 0.001,
"dataset": ["abilene", "geant"]
}

## How to use

Simple usage:
```bash
python main.py --model autotomo --unknown_r 0.1
```

Example usage:
```bash
python main.py --model autotomo --unknown_r 0.1 --dataset geant --loss_func l1norm --mode unobserved --epoch -1500
```

## Licence

Distributed under the MIT License. See [LICENCE](https://github.com/Y-debug-sys/AutoTomo/blob/main/LICENSE) for more information.
