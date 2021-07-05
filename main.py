import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import random
from contrast.SAGPool import SAGPool
from contrast.HGPSL import HGPSL
from model import Model
from utils import dataset_init, com_feature
from torch_geometric.datasets import TUDataset
from load_data import Dataset

parser = argparse.ArgumentParser(description='Multi-Scale Self-Attention Mixup for Graph Classification')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--exp_way', type=str, default='k_fold', help='random-split or cross-validation ')
parser.add_argument('--repetitions', type=int, default=10, help='number of repetitions (default: 10)')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--mixup', type=bool, default=True, help='whether use mixup')
parser.add_argument('--attention', type=bool, default=True, help='whether use self-attention')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--dataset', type=str, default='COLLAB', help='PROTEINS/DD/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=5000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--num_heads', type=int, default=8, help='alpha for mixup')
parser.add_argument('--alpha', type=int, default=0.1, help='alpha for mixup')
parser.add_argument('--Lev', type=int, default=2, help='level of transform (default: 2)')
parser.add_argument('--s', type=float, default=2, help='dilation scale > 1 (default: 2)')
parser.add_argument('--n', type=int, default=2,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
# HGPSL
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, optimizer, train_loader, test_loader, val_loader, i_fold):
    """

    :param train_loader:
    :param model: model
    :type optimizer: Optimizer

    """
    min_loss = 1e10
    patience = 0
    best_epoch = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            data = data.to(args.device)
            if args.mixup:
                out, y_b, lam = model(data, mixup=True, alpha=args.alpha)
                loss = lam * F.nll_loss(out, data.y) + (1 - lam) * F.nll_loss(out, y_b)
            else:
                out = model(data)
                loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        val_acc, val_loss = test_model(model, val_loader)
        test_acc, test_loss = test_model(model, test_loader)

        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
              'val_loss: {:.6f}'.format(val_loss), 'val_acc: {:.6f}'.format(val_acc),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))

        if val_loss < min_loss:
            torch.save(model.state_dict(), 'ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i_fold))
            print("Model saved at epoch{}".format(epoch))
            min_loss = val_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))

    return best_epoch


def test_model(model, loader):
    model.eval()
    correct = 0.
    test_loss = 0.
    for data in loader:
        with torch.no_grad():
            data = data.to(args.device)
            out = model(data)
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            test_loss += F.nll_loss(out, data.y, reduction='sum').item()
    acc = correct / len(loader.dataset)
    loss = test_loss / len(loader.dataset)
    return acc, loss


if __name__ == '__main__':
    acc = []
    loss = []
    setup_seed(args.seed)
    # Dataset initialization
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    if args.num_features == 0:
        dataset = com_feature(dataset)
        args.num_features = 1
    dataset, r = dataset_init(dataset, args)
    myDataset = Dataset(args, dataset)

    print(args)

    for i in range(args.repetitions):
        if args.exp_way == 'k_fold':
            train_loader, val_loader, test_loader = myDataset.kfold_split(i)
        elif args.exp_way == 'random_split':
            train_loader, val_loader, test_loader = myDataset.randomly_split()

        # Model initialization
        model = Model(args, r).to(args.device)
        # model = HGPSL(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Model training
        best_model = train(model, optimizer, train_loader, test_loader, val_loader, i)

        # Restore model for testing
        model.load_state_dict(torch.load('ckpt/{}/{}_fold_best_model.pth'.format(args.dataset, i)))
        test_acc, test_loss = test_model(model, test_loader)
        acc.append(test_acc)
        loss.append(test_loss)
        print('Test set results, best_epoch = {:.1f}  loss = {:.6f}, accuracy = {:.6f}'.format(best_model, test_loss,
                                                                                               test_acc))
    print(args)
    print('Total test set results, accuracy : {}'.format(acc))
    print('Average test set results, mean accuracy = {:.6f}, std = {:.6f}'.format(np.mean(acc), np.std(acc)))
