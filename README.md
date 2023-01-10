# Official Implementation: "AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography"

This is the github repoistory for an accurate and low-cos traffic estimation approach implemented in pytorch.

Paper: AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography

## Abstract

> Traffic matrix (TM) estimation is a high-cost but important task for network management. The-state-of-the-art methods use the emerging deep learning networks to learn the inverse mapping from link loads to OD flows, in order to estimate the complete TM using only the low-cost link loads. However all existing learning-based methods require a long-term training data including the historical instances of link loads and all OD flows in TM, whereas many networks only have instances of link loads and a part of OD flows due to the device limitations. In such cases, all current methods suffer from extremely low accuracy due to the deficient training target. In this paper, we demonstrate with a new proposal (named AutoTomo) that the inverse mapping from link loads to OD flows can be accurately learned even when the training instances cover only a few part of OD flows. Before introducing AutoTomo, we first parameterize the unknown flows and transform the target-deficient learning problem into a double-objective optimization problem integrating the tomography constrains. Then we propose the design of AutoTomo with two loss functions to optimize the two objectives through model training. Besides, we also propose an observation selection algorithm to guide the operator to monitor the most informative OD flows that can further improve the estimation accuracy. Finally, we evaluate our methods by two public traffic datasets Abilene and G´EANT. Comparing with four state-of-the-art learning based TM estimation methods, AutoTomo can averagely improve the accuracy of existing methods by about 15% with complete training data and 30% ∼ 53% with incomplete training data. Moreover, the results also demonstrate that our methods have fast testing time, which enables them to online TM estimation tasks.

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
- VAE https://github.com/MikeKalnt/VAE-TME
- FGSR https://github.com/udellgroup/Codes-of-FGSR-for-effecient-low-rank-matrix-recovery

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
"lr": 1e-2 if args.model=="mnetme" else 1e-3,
"dataset": ["abilene", "geant"],
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

## Author

- Yan Qiao -  [qiaoyan@hfut.edu.cn](qiaoyan:qiaoyan@hfut.edu.cn)
- Xinyu Yuan - [2022111103@mail.hfut.edu.cn](yuanxinyu:2022111103@mail.hfut.edu.cn) / [yxy5315@gmail.com](yuanxinyu:yxy5315@gmail.com)

## Citation

```
@InProceedings{Yanqiao23,
    author    = {Yan Qiao, Qui Wu, Xinyu Yuan.},
    title     = {AutoTomo: Accurate and Low-cost Traffic Estimator Integrating Network Tomography},
    booktitle = {International Conference on Distributed Computing Systems (ICDCS)},
    month     = {June},
    year      = {2023},
    pages     = {-}
}
```

## Licence

Distributed under the MIT License. See [LICENCE](https://github.com/Y-debug-sys/AutoTomo/blob/main/LICENSE) for more information.
