import argparse
from dataset import DatasetETHUCY
import util
import logging
import torch
from model.trajectory_model import TrajectoryModel
from torch.optim import Adam, lr_scheduler
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def run(args: util.Args, device):

    logger.info('**** data loading ******')
    train_dataset = DatasetETHUCY(args.data_dir, args.dataset, args.train_batch_size, False, args.end_centered,
                                  args.data_flip, args.data_scaling, args.obs_len, args.pred_len,
                                  args.split_temporal_interval, args.tree_degree, args.split_thea)

    test_dataset = DatasetETHUCY(args.data_dir, args.dataset, args.train_batch_size, True, args.end_centered,
                                 False, None, args.obs_len, args.pred_len,
                                 args.split_temporal_interval, args.tree_degree, args.split_thea)

    logger.info('**** model loading ******')
    model_args = util.ModelArgs  # You can change the arguments of model directly in the ModelArgs class
    model = TrajectoryModel(model_args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    reg_criterion = torch.nn.SmoothL1Loss().to(device)
    clf_criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.lr_scheduler == 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)
    if args.lr_scheduler == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_milestones[0])

    min_ade = 99
    min_fde = 99
    best_epoch = 0

    logger.info('**** model training ******')
    for epoch in range(args.epoch):
        total_loss, coarse_reg_loss, fine_reg_loss, clf_loss = train(args, model, optimizer, train_dataset,
                                                                     reg_criterion, clf_criterion, device)
        util.logging(
            f'dataset:{args.dataset} '
            f'epoch:{epoch} ',
            f'total_loss:{sum(total_loss) / len(total_loss)} ',
            f'coarse_reg_loss:{sum(coarse_reg_loss) / len(coarse_reg_loss)} ',
            f'fine_reg_loss:{sum(fine_reg_loss) / len(fine_reg_loss)} ',
            f'clf_loss:{sum(clf_loss) / len(clf_loss)} ',
            verbose=True,
            file_type='train_loss',
            append_log=True
        )

        ade, fde = test(args, model, test_dataset, device)

        util.logging(
            f'dataset:{args.dataset} '
            f'epoch:{epoch} ',
            f'ade:{ade} ',
            f'fde:{fde} ',
            verbose=True,
            file_type='ade_fde',
            append_log=True
        )
        if args.lr_scheduler == 1 or args.lr_scheduler == 0:
            scheduler.step()

        if min_fde + min_ade > ade + fde:
            min_fde = fde
            min_ade = ade
            best_epoch = epoch
            torch.save(model.state_dict(), args.checkpoint + args.dataset + '/model.pth')

        logger.info(f'dataset:{args.dataset}, curr_best_epoch:{best_epoch}, curr_min_ade:{min_ade},'
                    f' curr_min_fde:{min_fde}')

    logger.info(f'dataset:{args.dataset}, best_epoch:{best_epoch}, min_ade:{min_ade}, min_fde:{min_fde}')

    return


def get_train_loss(fine_trajs, gt_trajs, coarse_trajs, coarse_gt, path_score, closest_label, reg_criterion,
                   clf_criterion):
    fine_trajs = fine_trajs.reshape(gt_trajs.shape)
    coarse_trajs = coarse_trajs.reshape(coarse_gt.shape)
    coarse_reg_loss = reg_criterion(coarse_trajs, coarse_gt)
    fine_reg_loss = reg_criterion(fine_trajs, gt_trajs)
    clf_loss = clf_criterion(path_score, closest_label)
    loss = coarse_reg_loss + fine_reg_loss + clf_loss

    return loss, coarse_reg_loss, fine_reg_loss, clf_loss


def train(args: util.Args, model, optimizer, dataloader, reg_criterion, clf_criterion, device):
    model.train()
    train_loss_list = []
    coarse_reg_loss_list = []
    fine_reg_loss_list = []
    clf_loss_list = []

    for i, (trajs, masks, trees, coarse_gt, closest_label) in enumerate(
            zip(dataloader.trajectory_batches, dataloader.mask_batches, dataloader.traj_tree_batches,
                dataloader.coarse_gt_batches, dataloader.closest_branch_index_batches)):
        trajs = torch.FloatTensor(trajs).to(device)
        masks = torch.FloatTensor(masks).to(device)
        trees = torch.FloatTensor(trees).to(device)
        coarse_gt = torch.FloatTensor(coarse_gt).to(device)
        closest_label = torch.LongTensor(closest_label).to(device)

        obs_trajs = trajs[:, :args.obs_len, :]
        gt_trajs = trajs[:, args.obs_len:, :]

        optimizer.zero_grad()

        path_score, coarse_trajs, fine_trajs = model(obs_trajs, trees, coarse_gt, closest_label, masks, device)
        loss, coarse_reg_loss, fine_reg_loss, clf_loss = \
            get_train_loss(fine_trajs, gt_trajs, coarse_trajs, coarse_gt, path_score, closest_label, reg_criterion,
                           clf_criterion)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        coarse_reg_loss_list.append(coarse_reg_loss.item())
        fine_reg_loss_list.append(fine_reg_loss.item())
        clf_loss_list.append(clf_loss.item())

    return train_loss_list, coarse_reg_loss_list, fine_reg_loss_list, clf_loss_list


def test(args: util.Args, model, dataloader, device):

    model.eval()
    ade = 0
    fde = 0
    num_ped = 0
    num_trajs = 0

    for i, (trajs, masks, trees) in enumerate(zip(dataloader.trajectory_batches, dataloader.mask_batches,
                                                    dataloader.traj_tree_batches)):

        trajs = torch.FloatTensor(trajs).to(device)
        masks = torch.FloatTensor(masks).to(device)
        trees = torch.FloatTensor(trees).to(device)

        with torch.no_grad():
            obs_trajs = trajs[:, :args.obs_len, :]
            gt_trajs = trajs[:, args.obs_len:, :]

            num_trajs += obs_trajs.shape[0]

            pred_trajs, _ = model.predict(obs_trajs, trees, masks, args.num_k, device)
            min_ade, min_fde = util.get_ade_fde(pred_trajs, gt_trajs, args.num_k)
            ade += min_ade.item()
            fde += min_fde.item()

            num_ped += trajs.shape[0]

    ade = ade / num_ped
    fde = fde / num_ped

    return ade, fde


def main():

    logger.info('**** project args ******')
    parser = argparse.ArgumentParser()
    util.add_argument(parser)

    args: util.Args = parser.parse_args()
    util.init(args, logger)

    device = torch.device('cuda:' + str(args.gpu_num) if torch.cuda.is_available() and args.cuda else 'cpu')

    logger.info("device: {}".format(device))

    run(args, device)

    logger.info(f'Finished!')

if __name__ == '__main__':
    main()
