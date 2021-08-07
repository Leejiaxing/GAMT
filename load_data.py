import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, random_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from functools import reduce
from scipy import io
from utils import K_Fold
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Dataset:
    def __init__(self, args, dataset, split):
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.k_fold = args.repetitions
        self.k_fold_split = split

    def randomly_split(self, train_ratio=0.8, val_ratio=0.1):
        # trainset:validset:testset = 8:1:1
        num_training = int(len(self.dataset) * train_ratio)
        num_val = int(len(self.dataset) * val_ratio)
        num_test = len(self.dataset) - (num_training + num_val)

        # Randomly split dataset
        training_set, validation_set, test_set = random_split(self.dataset, [num_training, num_val, num_test])
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def kfold_split(self, test_index):
        assert test_index < self.k_fold
        valid_index = test_index - 1
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]

        train_mask = np.ones(len(self.dataset))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.dataset, train_split.tolist())
        valid_subset = Subset(self.dataset, valid_split.tolist())
        test_subset = Subset(self.dataset, test_split.tolist())

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader




class MultiCenterDataset:
    def __init__(self, args, root="data\multi-center"):
        self.batch_size = args.batch_size
        self.flie_list = os.listdir(root)
        self.flie_list.sort()
        self.threshold = 0.2
        self.fc = []

        self.class_dict = {
            "1": 0,
            "2": 1,
            "3": 1,
        }
        for label_files in self.flie_list:
            path = os.path.join(root, label_files)
            list = os.listdir(path)
            list.sort()
            sub_fc = []
            for file in list:
                a = 1
                label = torch.LongTensor([self.class_dict[file[0]]])
                data = pd.read_csv(os.path.join(path, file), header=None).T.to_numpy()
                subj_fc_adj = np.corrcoef(data)
                subj_fc_adj_list = subj_fc_adj.reshape((-1))
                thindex = int(self.threshold * subj_fc_adj_list.shape[0])
                thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex]]
                subj_fc_adj[subj_fc_adj < thremax] = 0
                subj_fc_adj[subj_fc_adj >= thremax] = 1
                feature = data[:, 0:170]
                edge_index, _ = dense_to_sparse(torch.from_numpy(subj_fc_adj.astype(np.int16)))
                sub_fc.append(Data(x=torch.from_numpy(feature).float(), edge_index=edge_index, y=label))
            self.fc.append(sub_fc)

    def split_data(self, test_index):
        train_set = []
        test_set = self.fc[test_index]
        valid_set = self.fc[test_index]
        self.fc.pop(test_index)
        for i in self.fc:
            train_set += i
        # train_mask = np.ones(len(self.dataset))
        # train_mask[test_split] = 0
        # train_mask[valid_split] = 0
        # train_split = train_mask.nonzero()[0]

        # train_subset = Subset(self.dataset, train_split)
        # valid_subset = Subset(self.dataset, valid_split)
        # test_subset = Subset(self.dataset, test_split)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader



