import numpy as np
import torch
import argparse
import os

from torch import nn, optim
from pathlib import Path
from net_params import AutoTomo, DBN, MNETME
from utils import train, test, auto_tomo_train, auto_tomo_test, EarlyStopping
from plot import plot_sp, plot_tm
from data_process import load_my_data


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment", add_help=False)

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience")
    parser.add_argument('--epoch', type=int, default=1000,
                        help="Epoch")
    parser.add_argument('--loss_func', type=str, default="l1norm",
                        choices=["l1norm", "mse"],
                        help="Loss function")
    parser.add_argument('--mode', type=str, default="all",
                        choices=["all", "unobserved"],
                        help="Show all flows in test_set or just unobserved flows in train_set")
    parser.add_argument('--dataset', type=str, default="abilene",
                        choices=["abilene", "geant"],
                        help="Dataset")
    parser.add_argument('--model', type=str, choices=['mnetme', 'dbn', 'autotomo'],
                        required=True, help="Choose Model")
    parser.add_argument('--unknown_r', type=float, choices=[i * 0.1 for i in range(7)],
                        required=True, help="Unknown rate")

    parser_args, _ = parser.parse_known_args()
    target_parser = argparse.ArgumentParser(parents=[parser])

    if parser_args.model == 'mnetme':
        target_parser.add_argument('--lr', type=float, default=1e-2,
                                   help="Learning rate")
        target_parser.add_argument('--hd', type=int, default=12,
                                   help="Hidden dimensions")
    elif parser_args.model in ['dbn', 'autotomo']:
        target_parser.add_argument('--lr', type=float, default=1e-3,
                                   help="Learning rate")
        target_parser.add_argument('--hd_1', type=int, default=80,
                                   help="Hidden dimensions 1st")
        target_parser.add_argument('--hd_2', type=int, default=100,
                                   help="Hidden dimensions 2nd")
        if parser_args.model == 'dbn':
            target_parser.add_argument('--hd_3', type=int, default=120,
                                       help="Hidden dimensions 3rd")

    args = target_parser.parse_args()
    return args


def main(args):
    if not Path("Model").exists():
        os.mkdir("Model")
    if not Path("Figure").exists():
        os.mkdir("Figure")
    check_point = os.path.join("Model", "best_model.pt")
    last_point = os.path.join("Model", "last_model.pt")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size, patience, epoch_num = args.batch_size, args.patience, args.epoch
    known_train_rate = 1 - args.unknown_r
    util_tag = False

    known_train_id, rm_tensor, train_flow_loader, test_flow_loader, train_link_loader, \
        test_link_loader, train_link_pinv_loader, test_link_pinv_loader, origin_train_flow_loader = \
        load_my_data(known_train_rate, batch_size, device, args.dataset)

    INPUT_SIZE = train_link_loader.dataset.shape[1]
    OUTPUT_SIZE = train_flow_loader.dataset.shape[1]

    if args.model == 'autotomo':
        util_tag = True
        model = AutoTomo(INPUT_SIZE, args.hd_1, args.hd_2, OUTPUT_SIZE).to(device)
    elif args.model == 'dbn':
        model = DBN(INPUT_SIZE, args.hd_1, args.hd_2, args.hd_3,
                    OUTPUT_SIZE, train_link_loader.dataset).to(device)
    elif args.model == 'mnetme':
        INPUT_SIZE = train_link_pinv_loader.dataset.shape[1]
        train_link_loader = train_link_pinv_loader
        test_link_loader = test_link_pinv_loader
        model = MNETME(INPUT_SIZE, args.hd, OUTPUT_SIZE).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=check_point)

    if args.loss_func == "l1norm":
        criteon = nn.L1Loss().to(device)
    elif args.loss_func == "mse":
        criteon = nn.MSELoss().to(device)

    if util_tag:
        """When AutoTomo is the seleted method."""

        rm_tensor = rm_tensor.to(device)
        # train
        model = auto_tomo_train(model, train_flow_loader, train_link_loader, criteon, optimizer,
                                early_stopping, epoch_num, known_train_id, check_point, last_point,
                                rm_tensor, device)

        # test and output results according to different mode
        if args.mode == "unobserved":
            unknown_train_id = np.setdiff1d(np.arange(OUTPUT_SIZE), known_train_id.cpu().numpy(), assume_unique=True)
            pred_train_flow = auto_tomo_test(model, origin_train_flow_loader, train_link_loader, criteon)
            pred_flow, real_flow = pred_train_flow[:, unknown_train_id], \
                origin_train_flow_loader.dataset.cpu().numpy()[:, unknown_train_id]
        elif args.mode == "all":
            pred_flow = auto_tomo_test(model, test_flow_loader, test_link_loader, criteon)
            real_flow = test_flow_loader.dataset.cpu().numpy()
    else:
        """When seleted method is DBN or MNETME."""

        # training
        model = train(model, train_flow_loader, train_link_loader, criteon, optimizer, early_stopping,
                      epoch_num, check_point, last_point)

        # test and output results according to different mode
        if args.mode == "unobserved":
            unknown_train_id = np.setdiff1d(np.arange(OUTPUT_SIZE), known_train_id.cpu().numpy(), assume_unique=True)

            pred_train_flow = test(model, origin_train_flow_loader, train_link_loader, criteon)
            pred_flow, real_flow = pred_train_flow[:, unknown_train_id], \
                origin_train_flow_loader.dataset.cpu().numpy()[:, unknown_train_id]
        elif args.mode == "all":
            pred_flow = test(model, test_flow_loader, test_link_loader, criteon)
            real_flow = test_flow_loader.dataset.cpu().numpy()

    """Plotting results"""

    plot_sp(real_flow, pred_flow, args.model, args.loss_func, os.path.join("Figure", args.mode + "_error_sp.jpg"))
    plot_tm(real_flow, pred_flow, args.model, args.loss_func, os.path.join("Figure", args.mode + "_error_tm.jpg"))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
