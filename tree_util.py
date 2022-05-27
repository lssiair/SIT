import numpy as np
import math
import matplotlib.pyplot as plt
import ipdb
import torch

def rotation_matrix(thea):
    return np.array([
        [np.cos(thea), -1 * np.sin(thea)],
        [np.sin(thea), np.cos(thea)]
    ])


def generating_tree(seq, dir_list, split_interval=4, degree=3):
    # seq [N n seq_len 2]
    # dir_list left, right, straight
    # return ---->  [N n*degree seq_len+interval 2]

    tree = np.zeros((seq.shape[0], seq.shape[1] * degree, seq.shape[2] + split_interval, seq.shape[-1]))
    for i in range(degree):
        curr_seq = seq
        curr_dir = np.expand_dims(dir_list[i], 2)  # N 1 1 2
        for j in range(split_interval):
            next_point = curr_seq[:, :, -1:] + curr_dir
            curr_seq = np.concatenate((curr_seq, next_point), axis=-2)
        tree[:, seq.shape[1] * i:seq.shape[1] * (i + 1)] = curr_seq
    return tree


def get_dir(seq, thea=12, degree=3, dir_interval=1):
    straight_dir = seq[:, :, -1] - seq[:, :, -dir_interval-1]  # N n 2
    straight_dir = straight_dir / dir_interval

    dir_list = [straight_dir]
    num_thea = int((degree - 1) / 2)

    for i in range(num_thea):
        th = (i + 1) * math.pi / thea
        left_dir = np.matmul(np.expand_dims(rotation_matrix(th), 0), np.transpose(straight_dir, (0, 2, 1)))
        right_dir = np.matmul(np.expand_dims(rotation_matrix(-th), 0), np.transpose(straight_dir, (0, 2, 1)))
        left_dir = np.transpose(left_dir, (0, 2, 1))
        right_dir = np.transpose(right_dir, (0, 2, 1))
        dir_list.append(left_dir)
        dir_list.append(right_dir)

    return dir_list


def tree_v3(traj_seq, degree, split_interval, pred_len=12, thea=12):
    # traj_seq [N obs_len 2]
    basic_tree = traj_seq  # N obs_len 2
    basic_tree = np.expand_dims(basic_tree, 1)  # N 1 obs_len 2
    dir_list = get_dir(basic_tree, thea=thea, degree=degree)  # split directions with the angle=pi/thea
    tree = generating_tree(basic_tree, dir_list, split_interval, degree)
    # angle= [4, 4]
    for i in range(1,  int(np.ceil(pred_len / split_interval))):
        tree = generating_tree(tree, dir_list, split_interval, degree)
        dir_list = get_dir(tree, 12 // (i + 1), degree=degree)
        # dir_list = get_dir(tree, angle[i-1], degree=degree)
        # dir_list = get_dir(tree, thea, degree=degree)

    return tree


def tree_build(traj_batches, split_interval=4, degree=3, pred_len=12, obs_len=8, thea=6):
    assert 1 <= split_interval <= pred_len
    tree_batches = []
    for b in traj_batches:
        obs_traj = b[:, :obs_len]
        tree = tree_v3(obs_traj, degree, split_interval, pred_len=pred_len, thea=thea)
        tree_batches.append(tree[:, :, obs_traj.shape[1]:b.shape[1]])  # truncating if over-length

    return tree_batches


def coarse_gt(full_trajs):

    # full_traj N pred_len+1 2
    obs_end_fut_traj = full_trajs[:, 7:]
    obs_traj = full_trajs[:, :8]
    selected_point = [0, 4, 8, 12]
    selected_seq = obs_end_fut_traj[:, selected_point]
    high_vel = selected_seq[:, 1:] - selected_seq[:, :-1]
    high_vel = high_vel / 4
    for i in range(12):
        if i < 4:
            next_point = obs_traj[:, -1:] + high_vel[:, 0:1]
            obs_traj = np.concatenate((obs_traj, next_point), axis=1)
        if 4 <= i < 8:
            next_point = obs_traj[:, -1:] + high_vel[:, 1:2]
            obs_traj = np.concatenate((obs_traj, next_point), axis=1)
        if 8 <= i < 12:
            next_point = obs_traj[:, -1:] + high_vel[:, 2:3]
            obs_traj = np.concatenate((obs_traj, next_point), axis=1)

    gt_ = obs_traj[:, 8:]

    return gt_


def tree_label(tree, traj_seq):

    closet_branch_index_batches = []
    coarse_gt_list = []

    for i in range(len(tree)):
        gt = coarse_gt(traj_seq[i])
        coarse_gt_list.append(gt)
        gt = np.expand_dims(gt, 1)  # N 1 pred_len 2
        tr = tree[i]

        distance_branch = np.linalg.norm(tr - gt, axis=-1)  # N n T
        # ade = np.mean(distance_branch, axis=-1)
        fde = distance_branch[:, :, -1]
        # distance = ade + fde
        # distance_branch = np.max(distance_branch, axis=-1)  # N n
        # one-hot label
        closet_branch_index = np.argmin(fde, axis=-1)
        closet_branch_index_batches.append(closet_branch_index)

    return closet_branch_index_batches, coarse_gt_list



def tree_build_iter(traj, split_interval=4, degree=3, pred_len=12, thea=12):


    traj = traj.permute(0, 2, 1)
    traj = traj[:, :, 2:]
    traj = traj.numpy()
    assert 1 <= split_interval <= pred_len
    obs_traj = traj
    tree = tree_v3(obs_traj, degree, split_interval, pred_len=pred_len, thea=thea)
    tree = tree - tree[:, :, 7:8]
    tree = tree[:, :, obs_traj.shape[1]:20]

    return torch.from_numpy(tree).float()


def tree_label(tree, traj_seq):
    # label_batches = []
    closest_branch_index_batches = []
    # closest_dir_index_batches = []
    coarse_gt_list = []
    interval = 4

    for i in range(len(tree)):
        # gt = traj_seq[i][:, 8:]
        gt = coarse_gt(traj_seq[i])
        coarse_gt_list.append(gt)
        gt = np.expand_dims(gt, 1)  # N 1 pred_len 2
        tr = tree[i]
        # dir = snowflake_[i]  # N n interval 2

        # distance_dir = np.linalg.norm(dir - gt[:, :, :interval], axis=-1)  # N n T
        # distance_dir = np.max(distance_dir, axis=-1)  # N n
        # one-hot label
        # closet_dir_index = np.argmin(distance_dir, axis=-1)   # N
        # closet_dir_index_batches.append(closet_dir_index)
        #
        # ade = np.linalg.norm(tr - gt, axis=-1).mean(axis=-1)  # N n
        # distance_ = np.exp(-ade)
        # dis_sum = np.sum(distance_, axis=1, keepdims=True)
        # soft_label = distance_ / dis_sum
        # min_fde_index = np.argmin(ade, axis=-1)
        # label_batches.append(min_fde_index)

        distance_branch = np.linalg.norm(tr - gt, axis=-1)  # N n T
        ade = np.mean(distance_branch, axis=1)
        fde = distance_branch[:, :, -1]
        # distance_branch = np.max(distance_branch, axis=-1)  # N n
        # one-hot label
        closet_branch_index = np.argmin(fde, axis=-1)
        # sec_fde_index = np.argsort(fde, axis=-1)[:, 1]
        closest_branch_index_batches.append(closet_branch_index)

    return closest_branch_index_batches, coarse_gt_list

def vis2(seq1):

    for i in range(seq1.shape[0]):
        plt.clf()
        for j in range(seq1.shape[1]):

            x1 = seq1[i, j, :, 0]
            y1 = seq1[i, j, :, 1]
            # x2 = seq2[i, :, 0]
            # y2 = seq2[i, :, 1]

            plt.plot(x1, y1, linestyle="-.", marker='.', color='red')
            # plt.plot(x2, y2, linestyle="-.", marker='.', color='green')

        plt.savefig('test_tree.png')
        ipdb.set_trace()



