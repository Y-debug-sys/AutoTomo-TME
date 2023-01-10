import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader


def add_unknown(feature_size, known_train_rate):

    known_train_num = int(np.ceil(feature_size * known_train_rate))
    id_rdm = torch.randperm(feature_size)
    known_train_id = id_rdm[0:known_train_num]
    id_rdm = torch.randperm(known_train_num)

    return known_train_id


def train_data(data, known_train_id):

    feature_size = data.shape[1]
    known_data_mean = torch.mean(data[:, known_train_id], dim=1).view(-1, 1)
    data_train_tensor = known_data_mean.repeat(1, feature_size)
    data_train_tensor[:, known_train_id] = data[:, known_train_id]

    return data_train_tensor


def load_my_data(known_train_rate, b_size, device, dataset_name):
    
    if dataset_name == "geant":
        base = "GeantData"
        data_file = os.path.join(base, "GeantTM.csv")
        rm_file = os.path.join(base, "geant_rm.csv")
        div_num = 10**7
        mult_num = 96
        train_day = 10*7
        test_day = 1*7
    
    if dataset_name == "abilene":
        base = "AbileneData"
        data_file = os.path.join(base, "AbileneTM.csv")
        rm_file = os.path.join(base, "abilene_rm.csv")
        div_num = 10**9
        mult_num = 288
        train_day = 15*7
        test_day = 1*7

    data = pd.read_csv(data_file, header=None)
    data.drop(data.columns[-1], axis=1, inplace=True)
    data_tensor = torch.from_numpy(data.values/div_num)
    data_tensor = data_tensor.float()

    data_size = data_tensor.shape[0]
    feature_size = data_tensor.shape[1]

    rm = pd.read_csv(rm_file, header=None)
    rm.drop(rm.columns[-1], axis=1, inplace=True)
    rm_tensor = torch.from_numpy(rm.values)
    rm_tensor = rm_tensor.float()

    rm_pinv_tensor = torch.linalg.pinv(rm_tensor)

    link_loads_tensor = data_tensor @ rm_tensor

    link_loads_pinv_tensor = link_loads_tensor @ rm_pinv_tensor

    known_train_id = add_unknown(feature_size, known_train_rate)
    data_train_tensor = train_data(data_tensor, known_train_id)
    data_train_tensor = data_train_tensor.float()

    train_size = int(train_day * mult_num)
    test_size = int(test_day * mult_num)

    id_seq = np.arange(data_size)
    train_id = id_seq[0:train_size]
    test_id = torch.randperm(test_size)
    test_id = test_id[:] + train_size

    origin_train_flow = data_tensor[train_id]

    train_flow = data_train_tensor[train_id]
    test_flow = data_tensor[test_id]

    train_link = link_loads_tensor[train_id]
    test_link = link_loads_tensor[test_id]

    train_link_pinv = link_loads_pinv_tensor[train_id]
    test_link_pinv = link_loads_pinv_tensor[test_id]

    train_flow_loader = DataLoader(dataset=train_flow.to(device), batch_size=b_size, shuffle=False)
    test_flow_loader = DataLoader(dataset=test_flow.to(device), batch_size=b_size, shuffle=False)
    train_link_loader = DataLoader(dataset=train_link.to(device), batch_size=b_size, shuffle=False)
    test_link_loader = DataLoader(dataset=test_link.to(device), batch_size=b_size, shuffle=False)
    train_link_pinv_loader = DataLoader(dataset=train_link_pinv.to(device), batch_size=b_size, shuffle=False)
    test_link_pinv_loader = DataLoader(dataset=test_link_pinv.to(device), batch_size=b_size, shuffle=False)
    origin_train_flow_loader = DataLoader(dataset=origin_train_flow.to(device), batch_size=b_size, shuffle=False)

    return known_train_id, rm_tensor, train_flow_loader, test_flow_loader, train_link_loader,\
          test_link_loader, train_link_pinv_loader, test_link_pinv_loader, origin_train_flow_loader