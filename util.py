from typing import Dict
import os
import subprocess
import random
import pickle
import torch
import numpy as np
import argparse


class Args:
    dataset = None
    epoch = None
    lr = None
    lr_scheduler = None
    lr_milestones = None
    lr_gamma = None
    obs_len = None
    pred_len = None
    train_batch_size = None
    test_batch_size = None
    seed = None
    gpu_num = None
    checkpoint = None
    data_dir = None
    log_dir = None
    cuda = None
    end_centered = None
    data_flip = None
    data_scaling = None

    # Arguments for the building of tree
    split_thea = None
    split_temporal_interval = None
    tree_degree = None
    num_k = None


class ModelArgs:
    # Arguments for model
    in_dim = 2
    obs_len = 8
    pred_len = 12
    hidden1 = 1024
    hidden2 = 256
    enc_dim = 64
    att_layer = 3
    tf = True  # teacher forcing
    out_dim = 2
    num_k = 20


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument('--dataset', type=str, default='eth', help='eth,hotel,univ,zara1,zara2,sdd')
    parser.add_argument('--data_dir', type=str,
                        default='/data0/liushuai/pec_data/')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--epoch', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=int, default=0, help='0:MultiStepLR, 1:CosineAnnealingLR, other numbers:None')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[50, 150, 250])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--train_batch_size', type=int, default=512,
                        help='256 or 512 for eth-ucy, 512 for sdd')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='256, 512 or 4096 for eth-ucy, 4096 for sdd')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_num', type=str, default='6')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/')
    parser.add_argument('--end_centered', action='store_true')
    parser.add_argument('--data_flip', action='store_true')
    parser.add_argument('--data_scaling', type=float, nargs='+', default=None)

    parser.add_argument('--split_thea', type=int, default=4)
    parser.add_argument('--split_temporal_interval', type=int, default=4)
    parser.add_argument('--tree_degree', type=int, default=3)

    parser.add_argument('--num_k', type=int, default=20)


def get_input_data(data_dict: Dict, key=None):

    try:
        return data_dict[key]
    except KeyError:
        print('KeyError')


args: Args = None
logger = None


def init(args_: Args, logger_):

    global args, logger

    args = args_
    logger = logger_

    # assert os.path.exists(args.checkpoint + args.dataset)
    assert os.path.exists(args.data_dir + 'test')
    assert os.path.exists(args.data_dir + 'train')

    if args.log_dir is None:
        args.log_dir = args.checkpoint + args.dataset

    # os.makedirs(args.checkpoint + args.dataset, exist_ok=True)
    # os.makedirs(args.log_dir, exist_ok=True)

    if os.path.exists(args.checkpoint + args.dataset):
        subprocess.check_output('rm -r {}'.format(args.checkpoint + args.dataset), shell=True, encoding='utf-8')
    os.makedirs(args.checkpoint + args.dataset, exist_ok=False)

    logger.info("*******" + ' args ' + "******")
    # args_dict = vars(args)
    # for key in args_dict:
    #     print("\033[32m" + key + "\033[0m", args_dict[key],  end='\t')
    # print('')
    logging(vars(args_), verbose=True, sep=' ', save_as_pickle=True, file_type=args.dataset + '.args')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def logging(*inputs, verbose=False, sep=' ', save_as_pickle=False, file_type='args', append_log=False):

    '''
        write something into log file
    :return:
    '''

    if verbose:
        print(*inputs, sep=sep)

    if not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, file_type)

    if save_as_pickle:
        with open(file, 'wb') as pickle_file:
            pickle.dump(*inputs, pickle_file)

    if append_log:
        with open(file, "a", encoding='utf-8') as fout:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)


def get_train_test_data(data_path, dataset_name, batch_size, is_test):

    if is_test:
        if dataset_name == 'sdd':
            return data_path + '/test' + "/social_" + dataset_name + "_test" + "_" + str(
                4096) + "_" + str(0) + "_" + str(100) + ".pickle"
        else:
            return data_path + '/test' + "/social_" + dataset_name + "_test" + "_" + str(
                    batch_size) + "_" + str(0) + "_" + str(50) + ".pickle"
    else:
        if dataset_name ==  'sdd':
            return data_path + '/train' + "/social_" + dataset_name + "_train" + "_" + str(
                512) + "_" + str(0) + "_" + str(100) + ".pickle"
        else:
            return data_path + '/train' + "/social_" + dataset_name + "_train" + "_" + str(
                    batch_size) + "_" + str(0) + "_" + str(50) + ".pickle"


def data_augmentation(data_, end_centered, is_flip, data_scaling):
    if end_centered:
        data_ = data_ - data_[:, 7:8]
    if is_flip:
        data_ = np.flip(data_, axis=-1).copy()
    if data_scaling is not None:
        data_[:, :, 0] = data_[:, :, 0] * data_scaling[0]
        data_[:, :, 1] = data_[:, :, 1] * data_scaling[1]

    return data_


def get_ade_fde(pred_trajs, gt_trajs, num_k):

    pred_trajs = pred_trajs.reshape(gt_trajs.shape[0], num_k, gt_trajs.shape[1], -1)
    gt_trajs = gt_trajs.unsqueeze(1)
    norm_ = torch.norm(pred_trajs - gt_trajs, p=2, dim=-1)
    ade_ = torch.mean(norm_, dim=-1)
    fde_ = norm_[:, :, -1]
    min_ade, _ = torch.min(ade_, dim=-1)
    min_fde, _ = torch.min(fde_, dim=-1)

    min_ade = torch.sum(min_ade)
    min_fde = torch.sum(min_fde)

    return min_ade, min_fde








