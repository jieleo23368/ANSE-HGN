import numpy as np
import torch
import time
import argparse
import os
import sys
from models.ANSEHGN import SubHIN


def setup_seed(seed):
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='HIN')
    parser.add_argument('--gpu', nargs='?', default='0')
    parser.add_argument('--model', nargs='?', default='ANSEHGN')
    parser.add_argument('--dataset', nargs='?', default='douban')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=20)

    #     parser.add_argument('--att_hid_units', type=int, default=64)
    parser.add_argument('--hid_units', type=int, default=128)
    parser.add_argument('--hid_units2', type=int, default=128)
    parser.add_argument('--out_ft', type=int, default=64)

    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--lamb_lp', type=float, default=1.0)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAtt', action='store_true', default=True)
    args = parser.parse_args()
    args.argv = sys.argv
    if args.gpu in {'0','1'}:
        torch.cuda.set_device(int(args.gpu))
    return args
    # return parser.parse_known_args()


def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)


def main():
    args= parse_args()
    printConfig(args)
    setup_seed(args.seed)
    if args.model == 'ANSEHGN':
        embedder = SubHIN(args)
    embedder.training()



if __name__ == '__main__':
    main()
