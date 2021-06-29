import os
import numpy as np
import torch
from torch.utils.data import Subset, random_split
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from math import floor
from functools import reduce
from scipy import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def K_Fold(k, len):
    split = []
    counter = 0
    block = len / k
    while counter < len - 0.5:
        c = counter + block
        split.append(list(range(floor(counter + 0.5), floor(c + 0.5))))
        counter = c
    all_index = list(range(len))
    np.random.shuffle(all_index)
    return [np.take(all_index, i, axis=0).tolist() for i in split]


class Dataset:
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.k_fold = args.repetitions
        self.k_fold_split = K_Fold(self.k_fold, len(self.dataset))

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
        valid_index = (test_index + 1) % self.k_fold
        test_split = self.k_fold_split[test_index]
        valid_split = self.k_fold_split[valid_index]
        train_split = reduce(lambda i, d: i if d[0] == test_index or d[0] == valid_index else i + d[1],
                             enumerate(self.k_fold_split), [])

        train_subset = Subset(self.dataset, train_split)
        valid_subset = Subset(self.dataset, valid_split)
        test_subset = Subset(self.dataset, test_split)

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(valid_subset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


class FSDataset:
    def __init__(self, root="E:\pythonproject\FCSC_pytorch\data_fMRI_DTI\\zhongda_data_fmri_dti"):
        label_list = os.listdir(root)
        label_list.sort()

        SC_adj_dir = "DTI_connectivity_count.mat"
        SC_feature_dir = "region_features_norm.mat"
        FC_dir = "RegionSeries.mat"
        threshold = 0.2
        self.sc = []
        self.fc = []
        self.class_dict = {
            "HC": 1,
            "MDD": 0,
        }
        for label_files in label_list:
            list = os.listdir(os.path.join(root, label_files))
            list.sort()
            label = torch.LongTensor([self.class_dict[label_files]])
            for files in list:
                subj_sc_adj_dir = os.path.join(root, label_files, files, SC_adj_dir)
                subj_sc_adj_data = io.loadmat(subj_sc_adj_dir)
                print("reading data " + subj_sc_adj_dir)
                subj_mat_sc_adj = subj_sc_adj_data['connectivity']
                subj_mat_sc_adj_list = subj_mat_sc_adj.reshape((-1))
                thindex = int(threshold * subj_mat_sc_adj_list.shape[0])
                thremax = subj_mat_sc_adj_list[subj_mat_sc_adj_list.argsort()[-1 * thindex]]
                subj_mat_sc_adj[subj_mat_sc_adj < thremax] = 0
                subj_mat_sc_adj[subj_mat_sc_adj >= thremax] = 1
                scedge_index, _ = dense_to_sparse(torch.from_numpy(subj_mat_sc_adj.astype(np.int16)))
                # adj.append(subj_mat_sc_adj)

                subj_sc_feature_dir = os.path.join(root, label_files, files, SC_feature_dir)
                subj_sc_feature_data = io.loadmat(subj_sc_feature_dir)
                print("reading data " + subj_sc_feature_dir)
                subj_mat_sc_feature = subj_sc_feature_data['region_features']
                # feature.append(subj_mat_sc_feature)
                self.sc.append(Data(x=torch.from_numpy(subj_mat_sc_feature).float(), edge_index=scedge_index,
                                    y=torch.tensor(label)))

                subj_fc_dir = os.path.join(root, label_files, files, FC_dir)
                subj_fc_data = io.loadmat(subj_fc_dir)
                print("reading data " + subj_fc_dir)
                subj_mat_fc = subj_fc_data['RegionSeries']
                subj_mat_fc_adj = np.corrcoef(np.transpose(subj_mat_fc))
                subj_mat_fc_adj_list = subj_mat_fc_adj.reshape((-1))
                thindex = int(threshold * subj_mat_fc_adj_list.shape[0])
                thremax = subj_mat_fc_adj_list[subj_mat_fc_adj_list.argsort()[-1 * thindex]]
                subj_mat_fc_adj[subj_mat_fc_adj < thremax] = 0
                subj_mat_fc_adj[subj_mat_fc_adj >= thremax] = 1
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(subj_mat_fc_adj.astype(np.int16)))

                subj_mat_fc_list = subj_mat_fc.reshape((-1))
                subj_mat_fc_new = (subj_mat_fc - min(subj_mat_fc_list)) / (
                        max(subj_mat_fc_list) - min(subj_mat_fc_list))
                subj_mat_fc_new = np.transpose(subj_mat_fc_new)

                self.fc.append(
                    Data(x=torch.from_numpy(subj_mat_fc_new).float(), edge_index=fcedge_index, y=torch.tensor(label)))

        # random.shuffle(self.fc)
        # random.shuffle(self.sc)
        self.k_fold = 10
        self.k_fold_split = K_Fold(self.k_fold, len(self.fc))
        self.choosen_dataset = self.sc

        self.train_iter_zd = [
            [28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100, 95, 30, 16, 5, 43, 48, 71, 6, 92, 106, 31, 85,
             101, 37,
             96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1, 69, 27, 109, 21, 98, 89, 18, 19, 63, 64, 99,
             61, 73, 3,
             41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75, 91,
             35, 29,
             84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 59, 12, 51, 100, 95, 30, 16, 5, 43, 48, 71, 6, 92, 106, 31,
             85, 101,
             37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1, 69, 27, 109, 21, 98, 89, 18, 19, 63, 64,
             99, 61,
             73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75,
             91, 35,
             29, 84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 6, 92, 106, 31, 85,
             101,
             37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1, 69, 27, 109, 21, 98, 89, 18, 19, 63, 64,
             99, 61,
             73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75,
             91, 35,
             29, 84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1, 69, 27, 109, 21, 98, 89, 18, 19, 63, 64, 99,
             61, 73,
             3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75, 91,
             35, 29,
             84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 69, 27, 109, 21, 98, 89, 18, 19, 63, 64,
             99, 61,
             73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75,
             91, 35,
             29, 84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1,
             61, 73,
             3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75, 91,
             35, 29,
             84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1,
             69, 27,
             109, 21, 98, 89, 18, 19, 63, 64, 99, 72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15, 0, 103, 14, 108, 76, 75,
             91, 35,
             29, 84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1,
             69, 27,
             109, 21, 98, 89, 18, 19, 63, 64, 99, 61, 73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 0, 103, 14, 108, 76, 75,
             91, 35,
             29, 84, 10, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1,
             69, 27,
             109, 21, 98, 89, 18, 19, 63, 64, 99, 61, 73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22,
             53, 66,
             68, 15, 23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49],
            [20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36, 28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86, 59, 12, 51, 100,
             95, 30,
             16, 5, 43, 48, 71, 6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52, 79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1,
             69, 27,
             109, 21, 98, 89, 18, 19, 63, 64, 99, 61, 73, 3, 41, 58, 25, 62, 104, 70, 4, 78, 72, 38, 24, 87, 46, 90, 22,
             53, 66,
             68, 15, 0, 103, 14, 108, 76, 75, 91, 35, 29, 84, 10]]
        self.test_iter_zd = [[20, 88, 45, 77, 83, 55, 32, 74, 107, 105, 36], [28, 9, 11, 47, 56, 80, 60, 8, 93, 50, 86],
                             [59, 12, 51, 100, 95, 30, 16, 5, 43, 48, 71],
                             [6, 92, 106, 31, 85, 101, 37, 96, 82, 94, 52],
                             [79, 34, 67, 54, 81, 42, 26, 102, 57, 39, 1],
                             [69, 27, 109, 21, 98, 89, 18, 19, 63, 64, 99],
                             [61, 73, 3, 41, 58, 25, 62, 104, 70, 4, 78], [72, 38, 24, 87, 46, 90, 22, 53, 66, 68, 15],
                             [0, 103, 14, 108, 76, 75, 91, 35, 29, 84, 10], [23, 44, 7, 17, 33, 40, 2, 97, 65, 13, 49]]

        self.train_iter_xx = [
            [22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51, 57, 7, 24, 17, 39, 40, 67, 85, 21,
             89, 2, 13,
             56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 31, 82, 12, 28, 26, 36, 88, 4, 51, 57, 7, 24, 17, 39, 40, 67, 85, 21,
             89, 2, 13,
             56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 57, 7, 24, 17, 39, 40, 67, 85, 21,
             89, 2, 13,
             56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             89, 2, 13,
             56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 89, 2, 13, 56, 33, 23, 70, 87, 54, 45, 75, 78, 73, 64, 25, 53, 35, 63, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 89, 2, 13, 56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 14, 72, 3,
             48, 76,
             65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 89, 2, 13, 56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78,
             73, 64,
             25, 53, 35, 63, 9, 37, 71, 66, 69, 6, 10, 32, 30, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 89, 2, 13, 56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78,
             73, 64,
             25, 53, 35, 63, 14, 72, 3, 48, 76, 65, 55, 83, 18, 8, 5, 52, 68, 49, 81, 59, 42, 20],
            [50, 47, 0, 84, 62, 74, 34, 11, 86, 22, 60, 41, 58, 1, 27, 77, 61, 15, 31, 82, 12, 28, 26, 36, 88, 4, 51,
             57, 7, 24,
             17, 39, 40, 67, 85, 21, 89, 2, 13, 56, 33, 23, 70, 87, 54, 29, 80, 43, 79, 38, 19, 44, 46, 16, 45, 75, 78,
             73, 64,
             25, 53, 35, 63, 14, 72, 3, 48, 76, 65, 55, 83, 18, 9, 37, 71, 66, 69, 6, 10, 32, 30]]
        self.test_iter_xx = [[50, 47, 0, 84, 62, 74, 34, 11, 86], [22, 60, 41, 58, 1, 27, 77, 61, 15],
                             [31, 82, 12, 28, 26, 36, 88, 4, 51], [57, 7, 24, 17, 39, 40, 67, 85, 21],
                             [89, 2, 13, 56, 33, 23, 70, 87, 54], [29, 80, 43, 79, 38, 19, 44, 46, 16],
                             [45, 75, 78, 73, 64, 25, 53, 35, 63], [14, 72, 3, 48, 76, 65, 55, 83, 18],
                             [9, 37, 71, 66, 69, 6, 10, 32, 30], [8, 5, 52, 68, 49, 81, 59, 42, 20]]

    def kFoldSplit(self, test_index):
        assert test_index < self.k_fold
        test_split = list(self.k_fold_split[test_index])
        train_split = reduce(lambda i, d: i if d[0] == test_index else i + list(d[1]), enumerate(self.k_fold_split), [])

        train_subset = Subset(self.choosen_dataset, train_split)
        test_subset = Subset(self.choosen_dataset, test_split)

        return DataLoader(train_subset, batch_size=8, shuffle=True), DataLoader(test_subset,
                                                                                batch_size=len(test_subset),
                                                                                shuffle=False)

    def kFoldSplit2(self, test_index):
        train_subset = Subset(self.choosen_dataset, self.train_iter_zd[test_index])
        test_subset = Subset(self.choosen_dataset, self.test_iter_zd[test_index])

        return DataLoader(train_subset, batch_size=8, shuffle=True), DataLoader(test_subset,
                                                                                batch_size=len(test_subset),
                                                                                shuffle=False)

    @property
    def meta(self):
        return [self.choosen_dataset[0].x.shape[1], 2]
