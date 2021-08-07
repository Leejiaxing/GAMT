import argparse
import os
import time

import numpy as np
import torch
from contrast.GCN import GCN
from model import Model, ModelHierarchical
from utils import dataset_init, com_feature, K_Fold
from train_test import test_model, train_model, setup_seed
from torch_geometric.datasets import TUDataset
from load_data import Dataset

parser = argparse.ArgumentParser(description='Multi-Scale Self-Attention Mixup for Graph Classification')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--exp_way', type=str, default='k_fold', help='random-split or cross-validation ')
parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions (default: 10)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--mixup', type=bool, default=True, help='whether use mixup')
parser.add_argument('--attention', type=bool, default=True, help='whether use self-attention')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--num_layers', type=int, default=3, help='the numbers of convolution layers')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='PROTEINS/DD/NCI1/NCI109/Mutagenicity/ENZYMES'
                                                                  '/IMDB-BINARY/PTC_FM/COLLAB')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=150, help='patience for early stopping')
parser.add_argument('--num_heads', type=int, default=8, help='alpha for mixup')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha for mixup')
parser.add_argument('--Lev', type=int, default=2, help='level of transform (default: 2)')
parser.add_argument('--s', type=float, default=2, help='dilation scale > 1 (default: 2)')
parser.add_argument('--n', type=int, default=2,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
args = parser.parse_args()


if __name__ == '__main__':
    acc = []
    loss = []
    setup_seed(args.seed)
    # Dataset initialization
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    split = K_Fold(args.repetitions, dataset)
    if args.num_features == 0:
        dataset = com_feature(dataset)
        args.num_features = 1
    dataset = dataset_init(dataset, args)
    myDataset = Dataset(args, dataset, split)

    print(args)

    for i in range(args.repetitions):
        if args.exp_way == 'k_fold':
            train_loader, val_loader, test_loader = myDataset.kfold_split(i)
        elif args.exp_way == 'random_split':
            train_loader, val_loader, test_loader = myDataset.randomly_split()

        # Model initialization
        model = Model(args).to(args.device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Model training
        best_model = train_model(args, model, optimizer, train_loader, val_loader, test_loader, i)

        # Restore model for testing
        model.load_state_dict(torch.load('ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i)))
        test_acc, test_loss = test_model(args, model, test_loader)
        acc.append(test_acc)
        loss.append(test_loss)
        print('Test set results, best_epoch = {:.1f}  loss = {:.6f}, accuracy = {:.6f}'.format(best_model, test_loss,
                                                                                               test_acc))
    print(args)
    print('Total test set results, accuracy : {}'.format(acc))
    print('Average test set results, mean accuracy = {:.6f}, std = {:.6f}'.format(np.mean(acc), np.std(acc)))
