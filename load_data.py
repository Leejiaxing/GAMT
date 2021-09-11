import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, random_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
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



