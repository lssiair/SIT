

import torch
import torch.nn as nn
from model.component import MLP
from model.component import SelfAttention
from util import ModelArgs


class TrajectoryModel(nn.Module):

    def __init__(self, args: ModelArgs):
        super(TrajectoryModel, self).__init__()

        in_dim = args.in_dim
        obs_len = args.obs_len
        pred_len = args.pred_len
        hidden1 = args.hidden1
        hidden2 = args.hidden2
        enc_dim = args.enc_dim
        att_layer = args.att_layer
        out_dim = args.out_dim

        self.obs_enc = nn.Sequential(
            MLP(in_dim*obs_len, hidden1),
            MLP(hidden1, hidden1),
            MLP(hidden1, hidden2),
            nn.Linear(hidden2, enc_dim)
        )

        # self attention for interaction
        self.int_att = nn.ModuleList(
            [SelfAttention(in_size=enc_dim, hidden_size=hidden2, out_size=enc_dim) for _ in range(att_layer)]
        )

        self.tree_enc = nn.Sequential(
            MLP(in_dim*pred_len, hidden1),
            MLP(hidden1, hidden1),
            MLP(hidden1, hidden2),
            nn.Linear(hidden2, enc_dim)
        )

        self.coarse_prediction = nn.Sequential(
            MLP(enc_dim*2, hidden1),
            MLP(hidden1, hidden1),
            MLP(hidden1, hidden2),
            nn.Linear(hidden2, out_dim*pred_len)
        )

        self.refining_enc = nn.Sequential(
            MLP(in_dim*pred_len, hidden1),
            MLP(hidden1, hidden1),
            MLP(hidden1, hidden2),
            nn.Linear(hidden2, enc_dim)
        )

        self.scoring_att = SelfAttention(in_size=enc_dim, hidden_size=hidden2, out_size=enc_dim)

        self.refining = nn.Sequential(
            MLP(enc_dim*2, hidden1),
            MLP(hidden1, hidden1),
            MLP(hidden1, hidden2),
            nn.Linear(hidden2, out_dim*pred_len)
        )

        self.output = nn.Linear(out_dim*pred_len, out_dim*pred_len)
        self.tf = args.tf

    def forward(self, obs_trajs, tree, coarse_gt, closest_label, mask, device):

        obs_trajs_ = obs_trajs.reshape(obs_trajs.shape[0], 1, -1)  # N 1 16
        tree = tree.reshape(tree.shape[0], tree.shape[1], -1)  # N n 24
        obs_enc = self.obs_enc(obs_trajs_)  # N 1 enc_dim

        obs_enc = obs_enc.permute(1, 0, 2)  # 1 N enc_dim

        for i in range(len(self.int_att)):
            int_mat = self.int_att[i](obs_enc, obs_enc, mask)
            obs_enc = obs_enc + torch.matmul(int_mat, obs_enc)
        obs_enc = obs_enc.permute(1, 0, 2)  # N 1 enc_dim

        tree_enc = self.tree_enc(tree)  # N n enc_dim
        path_score = self.scoring_att(obs_enc, tree_enc).squeeze()  # N n  # cross attention for classification

        ped_index = torch.arange(0, obs_trajs.shape[0]).to(device)
        closet_branch_enc = tree_enc[ped_index, closest_label]  # N enc_dim
        con_enc = torch.cat((obs_enc.squeeze(), closet_branch_enc), dim=-1)  # N enc_dim*2
        coarse_pred_traj = self.coarse_prediction(con_enc)  # N 24

        if self.tf:
            coarse_traj_ = coarse_gt.reshape(coarse_gt.shape)  # Teacher forcing
        else:
            coarse_traj_ = coarse_pred_traj  # without teacher forcing
        coarse_traj_ = coarse_traj_.reshape(coarse_traj_.shape[0], -1)
        coarse_enc = self.refining_enc(coarse_traj_)
        con_coarse_enc = torch.cat((obs_enc.squeeze(), coarse_enc), dim=-1)  # [N 128]
        refining_traj = self.refining(con_coarse_enc)
        predicted_traj = self.output(refining_traj)

        return path_score, coarse_pred_traj, predicted_traj

    def predict(self, obs_trajs, tree, mask, num_k, device):

        obs_trajs_ = obs_trajs.reshape(obs_trajs.shape[0], 1, -1)  # N 1 16
        tree = tree.reshape(tree.shape[0], tree.shape[1], -1)  # N n 24

        obs_enc = self.obs_enc(obs_trajs_)  # N 1 enc_dim
        tree_enc = self.tree_enc(tree)  # N n enc_dim

        obs_enc = obs_enc.permute(1, 0, 2)  # 1 N enc_dim
        for i in range(len(self.int_att)):
            int_mat = self.int_att[i](obs_enc, obs_enc, mask)
            obs_enc = obs_enc + torch.matmul(int_mat, obs_enc)
        obs_enc = obs_enc.permute(1, 0, 2)  # N 1 enc_dim

        path_score = self.scoring_att(obs_enc, tree_enc).squeeze()  # N n  # cross attention for classification
        top_k_indices = torch.topk(path_score, k=num_k, dim=-1).indices  # N num_k
        top_k_indices = top_k_indices.flatten()  # N*num_k
        ped_indices = torch.arange(0, obs_trajs.shape[0]).unsqueeze(1).to(device)  # N 1
        ped_indices = ped_indices.repeat(1, num_k).flatten()  # N*num_k
        selected_paths_enc = tree_enc[ped_indices, top_k_indices]  # N*num_k enc_dim
        selected_paths_enc = selected_paths_enc.reshape(tree_enc.shape[0], num_k, -1)
        obs_enc = obs_enc.repeat(1, selected_paths_enc.shape[1], 1)  # N num_k enc_dim
        con_enc = torch.cat((obs_enc, selected_paths_enc), dim=-1)  # N num_k enc_dim*2
        coarse_traj = self.coarse_prediction(con_enc)  # N num_k 24

        coarse_enc = self.refining_enc(coarse_traj)
        con_coarse_enc = torch.cat((obs_enc, coarse_enc), dim=-1)
        refining_traj = self.refining(con_coarse_enc)  # N num_k enc_dim
        predicted_traj = self.output(refining_traj)  # N num_k 24

        return predicted_traj, path_score
    # sdd  thea: 4 12 6
    # 9.71 17.26
    # 9.48 16.70
    # 9.44 16.62
    # 9.61 16.50
    # 9.62 16.19
    # 9.38 15.97
    # 9.25 15.57
    # 9.11 15.74
    # 9.12 15.63
    # 9.23 15.47
    # 9.09 15.54


    # eth
    # 0.41 0.62
    # 0.41 0.59 lr:0.001 thea:4 6 4


    # hotel
    # 0.15 0.29 lr:0.001 thea:4 6 4
    # 0.17 0.29 lr:0.001 thea:12 6 4
    # 0.18 0.28 lr:0.001 thea:12 12 12
    # 0.15 0.26 flip
    # 0.15 0.22 thea:12 6 4
    # 0.15 0.25 thea: 4 6 4
    # 0.14 0.25 thea: 4 6 4 [250]
    # 0.14 0.22 thea: 6 6 4


    # univ
    # 0.65 1.18 bs:512 thea:4 6 4
    # 0.27 0.47 bs:256 thea:4 6 4


    # zara1
    # 0.23 0.37 lr:0.003 thea:4 6 4
    # 0.21 0.36 lr：0.001 thea:4 6 4
    # 0.21 0.36
    # 0.20 0.34 thea:12 6 4
    # 0.19 0.33


    # zara2
    # 0.17 0.29 lr:0.003 thea:4 6 4
    # 0.16 0.29  lr:0.001
    # 0.16 0.30 12 6 4

