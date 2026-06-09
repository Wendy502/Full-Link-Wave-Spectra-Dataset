# setting.py
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn

def parse_opts():
    parser = argparse.ArgumentParser(description='Hs regression training')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=1)
    parser.add_argument('--niter', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--data_path', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--outf', default='./checkpoint')
    parser.add_argument('--log_path', default='./train_log')
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--decay_epoch', type=int, default=20, help='epoch to start linear lr decay')
    opt = parser.parse_known_args()[0]
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    opt.manual_seed = 42
    random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    cudnn.benchmark = True

    return opt
