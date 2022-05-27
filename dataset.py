import pickle

import numpy as np
from torch.utils import data
from util import get_train_test_data, data_augmentation

from tree_util import tree_build, tree_label


class DatasetETHUCY(data.Dataset):
    def __init__(self, data_path, dataset_name, batch_size, is_test, end_centered=True,
                data_flip=False, data_scaling=None, obs_len=8, pred_len=12,
                 split_interval=4, degree=3,  thea=6):

        'preprocessing for eth-ucy dataset'

        data_file = get_train_test_data(data_path, dataset_name, batch_size, is_test)

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        trajs, masks = data
        trajs_new = []

        for traj in trajs:
            t = np.array(traj)
            t = t[:, :, 2:4]
            t = data_augmentation(t, end_centered, data_flip, data_scaling)
            trajs_new.append(t)

        masks_new = []
        for mask in masks:
            masks_new.append(mask)

        traj_new = np.array(trajs_new)
        masks_new = np.array(masks_new)

        self.trajectory_batches = traj_new.copy()
        self.mask_batches = masks_new.copy()

        traj_tree = tree_build(traj_new.copy(), split_interval=split_interval, degree=degree, pred_len=pred_len, obs_len=obs_len, thea=thea)
        traj_tree = np.array(traj_tree)   # N n T 2
        self.traj_tree_batches = traj_tree.copy()

        # coarse ground truth
        if is_test is not True:
            closest_branch_index_batches, coarse_gt_list = \
                tree_label(traj_tree.copy(), traj_new.copy())
            closest_branch_index_batches = np.array(closest_branch_index_batches)
            coarse_gt_ = np.array(coarse_gt_list)
            self.closest_branch_index_batches = closest_branch_index_batches.copy()
            self.coarse_gt_batches = coarse_gt_.copy()

        print("Initialized dataloader for ucy-eth!")